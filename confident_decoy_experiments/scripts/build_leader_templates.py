#!/usr/bin/env python
"""`leader20` arm: the first DEPLOYABLE scheme -- top-20 leaders as templates, 5 seeds each, pool by ipTM.

THE METHOD (no ground truth needed at any step):
  1. free run, 250 poses                                      -- already have it
  2. greedy NMS on ipTM, same-basin = pose-pose DockQ >= 0.23  -- leaders, needs no native
  3. take the top 20 leaders; template each one separately     -- 20 runs x 5 seeds
  4. pool all resulting poses and take the highest interface pair_iptm
  5. that single pose is the method's answer

Every other arm that works is an ORACLE: dockqdonor picks its donor by DockQ, ifacelddt by lDDT vs
native, native50 uses the answer itself. The only previously-tested deployable arm, globalplddt, scores
0/15. This one uses nothing but the model's own outputs.

WHY 5 SEEDS, NOT 50: sub-sampling the existing dockqdonor arm shows the expected number of interfaces
rescued is flat from ~3 seeds (9.2 at K=5 vs 9.0 at K=50), and for five interfaces rescue probability
FALLS as seeds increase -- more sampling within one template condition lets the model find a wrong pose
it likes even better. So wide-and-shallow is cheaper AND possibly better than narrow-and-deep.
(50 seeds is still required to benchmark a SINGLE condition reliably -- ~3.7/25 verdicts churn at K=5 --
but this method's output is one pooled answer per interface, not a per-condition verdict.)

WHAT IT MEASURES: the conversion rate from "shortlist contains a good pose" (top-20 leaders: 23/26 =
88%) to "pooled ipTM selects a good pose" (unknown). Templating on the ORACLE best-docked pose rescues
only 9/25, so the ceiling is well below 88%.

NOTE: one template per run. Protenix cannot take several folds of the same chain at once -- it
de-duplicates template hits by the template structure's own sequence (report §12).

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_leader_templates.py [--topk 20]
Writes: artifacts/sweep_tmpl/leader20/<interface>_L<rank>.json,
        artifacts/sweep_a3m/leader20_*.a3m, artifacts/leader20_jobs.json, artifacts/leader20_index.json
"""
import argparse, csv, json, os, sys, glob, difflib, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gemmi
from cifconv import convert
from decoy_set import EXP, PRED, THRESH

ap = argparse.ArgumentParser()
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--seeds", type=int, default=5)
A = ap.parse_args()

JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
TMPL = f"{EXP}/artifacts/sweep_tmpl/leader20"; A3M = f"{EXP}/artifacts/sweep_a3m/leader20"
MATS = f"{EXP}/artifacts/pose_pose_dockq"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)
SEEDS = list(range(101, 101 + A.seeds))

TLC = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

sc = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
cache = json.load(open(CACHE_F))
TAKEN = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
_n = [0]


# Template ids MUST be exactly 4 characters, PDB-style. Protenix parses the a3m hit name with
# HitMetadata and rejects anything else -- "Invalid hit name format: g0024_A" -- producing
# "Found 0 templates" and a silently UNTEMPLATED run. This is what voided topdockq/topiptm/sposefv:
# their builders used a 4-digit counter (q1600+) which overflows to 5 characters past 999, while
# oracle's q900 stayed inside 4 and worked. Base-36 keeps 3 chars up to 46,655 ids.
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _b36(n, width=3):
    out = ""
    for _ in range(width):
        n, r = divmod(n, 36)
        out = B36[r] + out
    return out


def new_tid():
    while True:
        _n[0] += 1
        t = "w" + _b36(_n[0])
        assert len(t) == 4, f"tid {t} is not 4 characters"
        if t not in TAKEN:
            TAKEN.add(t); return t


def mut(a): return "A" if a != "A" else "G"


def native_seqs(pdb):
    path = f"/home/ebrahim/FoldBench/examples/ground_truths/{pdb}-assembly1.cif"
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_comp_id", "label_seq_id", "group_PDB")]
    seen, out = collections.defaultdict(set), collections.defaultdict(list)
    for a, comp, sid, g in zip(*cols):
        if g != "ATOM" or comp not in TLC or sid in seen[a]:
            continue
        seen[a].add(sid); out[a].append(TLC[comp])
    return {k: "".join(v) for k, v in out.items() if len(v) >= 4}


def best_chain(st, q):
    best, br = None, -1
    for c in st[0]:
        s = "".join(TLC[r.name] for r in c if r.name in TLC)
        if not s: continue
        r_ = difflib.SequenceMatcher(None, s, q, autojunk=False).ratio()
        if r_ > br: br, best = r_, c
    return best


def leaders(M, iptm, thresh=THRESH):
    order = np.argsort(-iptm); alive = np.ones(len(iptm), bool); out = []
    for i in order:
        if not alive[i]: continue
        out.append(int(i)); alive &= ~(M[i] >= thresh); alive[i] = False
    return out


jobs, index, skipped = [], {}, []
for lab in sorted(k for k, v in sc.items()
                  if v["free_status"] == "confident_decoy" and k not in {"7z2m_AJ", "8udz_AC"}):
    npy = f"{MATS}/dockq_{lab}_seeds101-150.npy"; idx = f"{MATS}/dockq_{lab}_seeds101-150_index.csv"
    if not (os.path.exists(npy) and os.path.exists(idx)):
        skipped.append(f"{lab}: no pose-pose matrix"); continue
    r = list(csv.DictReader(open(idx)))
    dq = np.array([float(x["dockq_gt"]) for x in r])
    ip = np.array([float(x["iptm"]) for x in r])
    L = leaders(np.load(npy), ip)[:A.topk]

    v = sc[lab]; pdb = v["pdb"]
    base = json.load(open(f"{JDIR}/{pdb}-final-updated.json"))
    natseq = native_seqs(pdb)
    parts = v["chains"].split("-")
    c1, c2 = (parts[0], "-".join(parts[1:])) if len(parts) > 2 else (parts[0], parts[1])

    index[lab] = {"pdb": pdb, "interface": [c1, c2], "n_leaders_used": len(L), "leaders": []}
    for rank, p in enumerate(L, start=1):
        s, k = int(r[p]["seed"]), int(r[p]["sample"])
        cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
        if not os.path.exists(cif):
            skipped.append(f"{lab} L{rank}: cif missing"); continue
        obj = json.loads(json.dumps(base[0] if isinstance(base, list) else base))
        qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
        tid = new_tid(); cache.setdefault(tid, {"release_date": "2000-01-01"})
        open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(cif).read(), tid))
        st_ = gemmi.read_structure(cif); st_.setup_entities()

        pi, ok = 0, True
        for e in obj["sequences"]:
            if "proteinChain" not in e:
                continue
            q = qseqs[pi]; mc = best_chain(st_, q)
            if mc is None:
                ok = False; break
            L_ = len(q); hs = list(q)
            for pos in (L_ // 3, 2 * L_ // 3):
                hs[pos] = mut(hs[pos])
            a3mp = f"{A3M}/leader20_{lab}_L{rank:02d}_q{pi}.a3m"
            open(a3mp, "w").write(
                f">query\n{q}\n>{tid}_{mc.name}/1-{L_} [subseq from] mol:protein length:{L_}  leader20\n"
                f"{''.join(hs)}\n")
            e["proteinChain"]["templatesPath"] = a3mp
            pi += 1
        if not ok:
            skipped.append(f"{lab} L{rank}: no model chain matched"); continue

        name = f"{pdb}_{c1}{c2}_L{rank:02d}".replace("-", "")
        # MUST set this: Protenix names the output directory AND every cif after obj["name"].
        # Left unset it stays as the base json's name (the bare pdb id), so all 20 template runs of
        # one pdb write into the SAME directory and overwrite each other, while the scheduler --
        # which looks for "{name}_summary_confidence_sample_4.json" -- records 0/5 seeds for jobs
        # that actually ran.
        obj["name"] = name
        jp = f"{TMPL}/{lab}_L{rank:02d}.json"
        json.dump([obj], open(jp, "w"))
        jobs.append({"run": "leader20", "pdb": pdb, "interface": lab, "leader_rank": rank,
                     "json": jp, "name": name, "seeds": SEEDS, "set": "decoy"})
        index[lab]["leaders"].append({"rank": rank, "seed": s, "sample": k,
                                      "iptm": round(float(ip[p]), 3),
                                      "dockq_gt": round(float(dq[p]), 3),
                                      "acceptable": bool(dq[p] >= THRESH), "name": name, "tid": tid})
    n_ok = sum(1 for x in index[lab]["leaders"] if x["acceptable"])
    print(f"{lab:11} {len(index[lab]['leaders']):2d} leaders, {n_ok} acceptable among them, "
          f"first acceptable at rank "
          f"{next((x['rank'] for x in index[lab]['leaders'] if x['acceptable']), None)}", flush=True)

json.dump(cache, open(CACHE_F, "w"))
json.dump(jobs, open(f"{EXP}/artifacts/leader20_jobs.json", "w"), indent=1)
json.dump(index, open(f"{EXP}/artifacts/leader20_index.json", "w"), indent=1)
print(f"\nBUILT {len(jobs)} jobs across {len(index)} interfaces "
      f"({A.topk} templates x {A.seeds} seeds each = {len(jobs)*A.seeds} seeds total)")
for s_ in skipped[:10]:
    print("  SKIP " + s_)
