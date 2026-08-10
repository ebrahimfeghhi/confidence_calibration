#!/usr/bin/env python
"""HILL-CLIMB on 8che: each round's best poses (by ipTM) become the next round's templates.

THE IDEA (user's, 2026-08-07). Every run so far is independent -- run N learns nothing from run
N-1. Here the search is iterative: generate, rank by ipTM, template on the winners, generate again.
If templating localises the search around the template's fold, then climbing on ipTM should walk the
pool toward better folds without ever seeing a label.

DESIGN (as specified by the user):
  round 0    : 25 FREE seeds, no injected template            -> 125 poses
  round 1-39 : top 5 poses by ipTM from the CUMULATIVE pool, one arm each, 5 seeds per arm
               -> 5 arms x 5 seeds x 5 samples = 125 poses per round
  40 rounds x 25 seeds = 1000 seeds / 5000 poses total.

  Parent selection is PLAIN top-5 by ipTM over every pose generated so far -- no diversity filter,
  no DockQ anywhere in the loop. The user chose plain over an NMS-filtered variant deliberately, to
  test the idea as stated. Cumulative rather than per-round was chosen on the measurement that once a
  near-native pose exists ipTM ranks it above the decoy (allpose8che10: near-native poses 0.68-0.73,
  the confident decoy 0.44-0.53), so elitism should move toward good poses rather than lock onto the
  decoy -- but the decoy CAN dominate the early rounds, and the round log is what shows that.

  Seeds are unique WITHIN this experiment (never reused across rounds) but are NOT held disjoint from
  8che seeds used in earlier experiments -- the user's explicit call. Round 0 will therefore overlap
  the existing free pool, which gives a free reproducibility check.

RANKING SIGNAL. score(pose) = mean(pair_iptm[A,E], pair_iptm[B,E]). 8che has two confident-decoy
interfaces and 137_3 moved both together, so averaging avoids optimising one at the other's expense.
Both interfaces are logged separately regardless, so a BE-only variant can be re-derived offline
without re-running anything.

CHAIN INDEXING. chain_pair_iptm is indexed by model chain letter (index = ord(letter)-65), and OST
maps native A-E -> model A-E and native B-E -> model C-E. Model chain order is
['A','B','C','D','E','F'] in both free and templated predictions. The order is ASSERTED against a
freshly-read cif once per round rather than assumed; if it ever changes, the run stops instead of
silently ranking on the wrong interface.

DOCKQ IS FOR THE LOG ONLY. Every round is scored with OST so the trajectory is readable, but no
DockQ value enters parent selection. The algorithm is label-free.

TRAPS GUARDED (each has burned a previous arm -- see build_allpose_templates.py):
  - 4-CHARACTER template ids. Protenix raises `Invalid hit name format` otherwise and logs
    `Found 0 templates`, producing a silently UNTEMPLATED run that still emits plausible poses.
    tids here are h000-h199, asserted 4 chars and asserted absent from /home/ebrahim/mmcif.
  - obj["name"] unique per job, else every arm writes into one output directory.
  - the a3m hit must differ from the query (2 mutated residues), else it is read as an MSA row.
  - round 0 must keep the NATURAL templatesPath (real hmmsearch templates); only rounds >=1 replace
    it. A round-0 that silently dropped the natural MSA/templates would not be the free baseline.
  - after every templated round, the round's logs are checked for `Found 0 templates` and the run
    ABORTS on any hit. A null round would climb on nothing.

RESUMABLE: a round whose 25 seeds already have sample_4 confidence JSONs is not regenerated; the
driver re-harvests it and moves on. Safe to kill and relaunch.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/hillclimb8che.py [--rounds 40] [--gpus 0,1,2,3]
Writes: artifacts/sweep_out/hillclimb8che/...      predictions
        artifacts/hillclimb8che_pool.json          every pose with its ipTM and DockQ
        artifacts/hillclimb8che_rounds.json        per-round parents + trajectory
        artifacts/hillclimb8che_ost/               per-pose OST cache
"""
import argparse, json, os, sys, glob, difflib, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemmi
from cifconv import convert
from decoy_set import EXP, OST, ENV, GT
from build_dockq_templates import native_seqs, best_chain, mut

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--seeds-per-round", type=int, default=25)
ap.add_argument("--parents", type=int, default=5)
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--workers", type=int, default=16)
A = ap.parse_args()

RUN = "hillclimb8che"
PDB = "8che"
NSAMP = 5
SEEDS_PER_ARM = A.seeds_per_round // A.parents          # 25 / 5 = 5
JDIR = "/home/ebrahim/Protenix/foldbench_abag/jsons"
ART = f"{EXP}/artifacts"
TMPL = f"{ART}/sweep_tmpl/{RUN}"
A3M = f"{ART}/sweep_a3m"
OUTROOT = f"{ART}/sweep_out/{RUN}"
LOGD = f"{ART}/{RUN}_logs"
CACHEO = f"{ART}/{RUN}_ost"
CACHE_F = "/home/ebrahim/common/release_date_cache.json"
PY = "/home/ebrahim/Protenix/.venv/bin/python"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for d in (TMPL, A3M, OUTROOT, LOGD, CACHEO):
    os.makedirs(d, exist_ok=True)

# model chain letters per native interface (see CHAIN INDEXING in the docstring)
IFACE_MODEL = {"8che_AE": ("A", "E"), "8che_BE": ("C", "E")}
CHAIN_ORDER = ["A", "B", "C", "D", "E", "F"]

base = json.load(open(f"{JDIR}/{PDB}-final-updated.json"))
BASE = base[0] if isinstance(base, list) else base
natseq = native_seqs(PDB)

# one shuffled seed stream, consumed 25 per round; unique within this experiment
SEED_STREAM = random.Random(0).sample(range(1, 100000), A.rounds * A.seeds_per_round)


def log(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def iptm_of(conf_path):
    """{iface: pair_iptm} from a summary_confidence json, by model chain letter."""
    M = json.load(open(conf_path))["chain_pair_iptm"]
    out = {}
    for lab, (c1, c2) in IFACE_MODEL.items():
        i, j = ord(c1) - 65, ord(c2) - 65
        if not (0 <= i < len(M) and 0 <= j < len(M)):
            raise SystemExit(f"chain_pair_iptm too small ({len(M)}) for {lab} in {conf_path}")
        out[lab] = M[i][j]
    return out


def assert_chain_order(cif):
    st = gemmi.read_structure(cif)
    st.setup_entities()
    names = [c.name for c in st[0]]
    if names[:len(CHAIN_ORDER)] != CHAIN_ORDER:
        raise SystemExit(f"ABORT: model chain order changed: {names} != {CHAIN_ORDER} in {cif}\n"
                         "ipTM indexing would silently rank the wrong interface.")


def dockq_of(cif, tag):
    """{iface: dockq} for one pose, via the project-standard OST call. Log only."""
    o = f"{CACHEO}/{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        subprocess.run(["taskset", "-c", "0-63", OST, "compare-structures", "-m", cif,
                        "-r", f"{GT}/{PDB}-assembly1.cif", "-o", o, "--fault-tolerant",
                        "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq"],
                       env=ENV, capture_output=True, timeout=900)
    try:
        d = json.load(open(o))
    except Exception:
        return {}
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    got = {}
    for lab, (c1, c2) in {"8che_AE": ("A", "E"), "8che_BE": ("B", "E")}.items():
        for i, t in enumerate(ifs):
            if {t[0], t[1]} == {c1, c2} and i < len(qs):        # native names, BY NAME
                got[lab] = qs[i]
                break
    return got


def build_template_job(pose, tid, name, seeds):
    """One arm: template all three chains of 8che from `pose`, run `seeds`."""
    assert len(tid) == 4, f"template id {tid!r} is not 4 characters"
    taken = {os.path.basename(p)[:-4] for p in glob.glob("/home/ebrahim/mmcif/*.cif")}
    assert tid not in taken, f"template id {tid} already registered in /home/ebrahim/mmcif"
    cache = json.load(open(CACHE_F))
    cache.setdefault(tid, {"release_date": "2000-01-01"})
    json.dump(cache, open(CACHE_F, "w"))
    open(f"/home/ebrahim/mmcif/{tid}.cif", "w").write(convert(open(pose["cif"]).read(), tid))

    obj = json.loads(json.dumps(BASE))
    qseqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    st = gemmi.read_structure(pose["cif"])
    st.setup_entities()
    pi, nch = 0, 0
    for e in obj["sequences"]:
        if "proteinChain" not in e:
            continue
        q = qseqs[pi]
        mc = best_chain(st, q)
        if mc is None:
            raise SystemExit(f"ABORT: no model chain matched query {pi} of {pose['cif']}")
        L = len(q)
        hs = list(q)
        for p in (L // 3, 2 * L // 3):                  # hit MUST differ from query
            hs[p] = mut(hs[p])
        a3mp = f"{A3M}/{RUN}_{name}_q{pi}.a3m"
        open(a3mp, "w").write(
            f">query\n{q}\n>{tid}_{mc.name}/1-{L} [subseq from] mol:protein length:{L}  {RUN}\n"
            f"{''.join(hs)}\n")
        e["proteinChain"]["templatesPath"] = a3mp
        pi += 1
        nch += 1
    assert nch == 3, f"expected 3 templated chains, got {nch}"
    obj["name"] = name                                  # unique per job -- see docstring
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"))
    return {"run": RUN, "pdb": PDB, "interface": name, "json": jp, "name": name,
            "seeds": seeds, "set": "hillclimb", "parent": pose["tag"], "tid": tid}


def build_free_job(name, seeds):
    """Round 0: untemplated. Keeps the NATURAL templatesPath -- see docstring."""
    obj = json.loads(json.dumps(BASE))
    nat = [e["proteinChain"].get("templatesPath") for e in obj["sequences"] if "proteinChain" in e]
    assert all(p and "hmmsearch" in p for p in nat), \
        f"round 0 must keep the natural hmmsearch templates, got {nat}"
    obj["name"] = name
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"))
    return {"run": RUN, "pdb": PDB, "interface": name, "json": jp, "name": name,
            "seeds": seeds, "set": "hillclimb", "parent": None, "tid": None}


def run_jobs(jobs, rnd):
    """Generate, then verify the intervention actually applied."""
    jf = f"{ART}/{RUN}_r{rnd:02d}_jobs.json"
    json.dump(jobs, open(jf, "w"), indent=1)
    want = sum(len(j["seeds"]) for j in jobs)
    have = sum(1 for j in jobs for s in j["seeds"]
               if os.path.exists(f"{OUTROOT}/{RUN}/{j['name']}/seed_{s}/predictions/"
                                 f"{j['name']}_summary_confidence_sample_{NSAMP - 1}.json"))
    if have >= want:
        log(f"round {rnd}: already complete ({have}/{want} seeds), re-harvesting")
    else:
        subprocess.run([PY, f"{SCRIPTS}/sweep_sched.py", "--jobs", jf, "--out", OUTROOT,
                        "--logs", LOGD, "--status", f"{ART}/{RUN}_r{rnd:02d}_status.json",
                        "--gpus", A.gpus, "--n-sample", str(NSAMP)], check=True)
    # null-template audit: a templated round that loaded nothing would climb on noise
    if rnd > 0:
        nulls = sum(1 for j in jobs
                    for l in open(f"{LOGD}/{RUN}_{j['name']}.log", errors="ignore")
                    if "Found 0 templates" in l)
        if nulls:
            raise SystemExit(f"ABORT round {rnd}: {nulls} chain-loads reported 'Found 0 templates'")
    got = sum(1 for j in jobs for s in j["seeds"]
              if os.path.exists(f"{OUTROOT}/{RUN}/{j['name']}/seed_{s}/predictions/"
                                f"{j['name']}_summary_confidence_sample_{NSAMP - 1}.json"))
    if got < want:
        raise SystemExit(f"ABORT round {rnd}: {got}/{want} seeds produced output")
    return jobs


def harvest(jobs, rnd):
    """Every pose of this round with its ipTM (ranking) and DockQ (log only)."""
    poses = []
    for j in jobs:
        for s in j["seeds"]:
            p = f"{OUTROOT}/{RUN}/{j['name']}/seed_{s}/predictions"
            for k in range(NSAMP):
                cif = f"{p}/{j['name']}_sample_{k}.cif"
                conf = f"{p}/{j['name']}_summary_confidence_sample_{k}.json"
                if not (os.path.exists(cif) and os.path.exists(conf)):
                    raise SystemExit(f"ABORT: missing pose {cif}")
                it = iptm_of(conf)
                poses.append({"tag": f"r{rnd:02d}_{j['name']}_{s}_{k}", "cif": cif,
                              "round": rnd, "arm": j["name"], "seed": s, "sample": k,
                              "parent": j.get("parent"),
                              "iptm_AE": round(it["8che_AE"], 4),
                              "iptm_BE": round(it["8che_BE"], 4),
                              "iptm": round((it["8che_AE"] + it["8che_BE"]) / 2, 4)})
    assert_chain_order(poses[0]["cif"])                 # per round, not assumed
    with ThreadPoolExecutor(A.workers) as ex:
        dqs = list(ex.map(lambda x: dockq_of(x["cif"], x["tag"]), poses))
    for p, d in zip(poses, dqs):
        p["dockq_AE"] = round(d.get("8che_AE", float("nan")), 3)
        p["dockq_BE"] = round(d.get("8che_BE", float("nan")), 3)
    return poses


POOLF = f"{ART}/{RUN}_pool.json"
ROUNDF = f"{ART}/{RUN}_rounds.json"
pool = json.load(open(POOLF)) if os.path.exists(POOLF) else []
rounds = json.load(open(ROUNDF)) if os.path.exists(ROUNDF) else []
done_rounds = {r["round"] for r in rounds}

log(f"{RUN}: {A.rounds} rounds x {A.seeds_per_round} seeds "
    f"({A.parents} parents x {SEEDS_PER_ARM} seeds from round 1), GPUs {A.gpus}")

for rnd in range(A.rounds):
    if rnd in done_rounds:
        log(f"round {rnd}: already recorded, skipping")
        continue
    seeds = SEED_STREAM[rnd * A.seeds_per_round:(rnd + 1) * A.seeds_per_round]
    assert len(set(seeds)) == len(seeds) == A.seeds_per_round

    if rnd == 0:
        jobs = [build_free_job(f"{PDB}_hc_r00_free", seeds)]
        parents = []
    else:
        parents = sorted(pool, key=lambda p: -p["iptm"])[:A.parents]     # PLAIN top-5, cumulative
        jobs = []
        for i, par in enumerate(parents):
            tid = f"h{(rnd - 1) * A.parents + i:03d}"
            nm = f"{PDB}_hc_r{rnd:02d}_p{i}"
            jobs.append(build_template_job(par, tid, nm,
                                           seeds[i * SEEDS_PER_ARM:(i + 1) * SEEDS_PER_ARM]))
        log(f"round {rnd} parents: " + ", ".join(
            f"{p['tag']} ipTM {p['iptm']:.3f} (DockQ BE {p['dockq_BE']:.3f})" for p in parents))

    run_jobs(jobs, rnd)
    new = harvest(jobs, rnd)
    pool.extend(new)

    bb = max(p["dockq_BE"] for p in pool)
    ba = max(p["dockq_AE"] for p in pool)
    sel = max(pool, key=lambda p: p["iptm"])
    rec = {"round": rnd, "seeds": seeds, "n_new_poses": len(new), "pool_size": len(pool),
           "parents": [{"tag": p["tag"], "iptm": p["iptm"], "dockq_BE": p["dockq_BE"],
                        "dockq_AE": p["dockq_AE"]} for p in parents],
           "round_best_iptm": round(max(p["iptm"] for p in new), 4),
           "round_best_dockq_BE": max(p["dockq_BE"] for p in new),
           "pool_best_dockq_BE": bb, "pool_best_dockq_AE": ba,
           "pool_sel_tag": sel["tag"], "pool_sel_iptm": sel["iptm"],
           "pool_sel_dockq_BE": sel["dockq_BE"], "pool_sel_dockq_AE": sel["dockq_AE"]}
    rounds.append(rec)
    json.dump(pool, open(POOLF, "w"))
    json.dump(rounds, open(ROUNDF, "w"), indent=1)
    log(f"round {rnd} done: pool {len(pool)} poses | pool best DockQ BE {bb:.3f} AE {ba:.3f} | "
        f"ipTM-selected DockQ BE {sel['dockq_BE']:.3f} (ipTM {sel['iptm']:.3f})")

log("ALL ROUNDS DONE")
log(f"final: pool {len(pool)} poses, best DockQ BE {max(p['dockq_BE'] for p in pool):.3f}, "
    f"ipTM-selected BE {max(pool, key=lambda p: p['iptm'])['dockq_BE']:.3f}")
