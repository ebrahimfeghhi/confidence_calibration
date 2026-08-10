#!/usr/bin/env python
"""Does removing the MSA help on OTHER dockqdonor interfaces, or only on 8che?

On 8che, templating on donor 137_3 with the MSA replaced by a single query sequence took seeds
producing a near-native pose from 6/10 to 9/10 (masked, dropout on; see unmask_factorial_scores.json).
That is one donor on one interface. This repeats the condition on every interface dockqdonor rescues.

DESIGN. For each interface, take dockqdonor's OWN donor and its OWN template json, replace both MSA
paths with a 1-sequence a3m (the query), and run 10 seeds. Everything else is stock: MASKED (normal
multichain_mask, from /home/ebrahim/Protenix), dropout at the default 0.4.

  SEEDS = the donor's OWN seed + 9 more drawn at random from 101-150 (rng-seeded, reproducible).
  Every seed lies inside dockqdonor's own 101-150 range, so the with-MSA comparison is the existing
  dockqdonor run restricted to those same 10 seeds -- exactly seed-matched, template-matched and
  donor-matched, with no new baseline runs. Including the own seed matches how allpose8che10 and the
  unmask factorial were built (137 was in the factorial's set), so no donor is judged only on foreign
  seeds. A fixed 101-110 block was rejected: it contains NONE of the seven donors' own seeds
  (124,124,120,137,115,122,132).

  Interfaces sharing a pdb+donor are run ONCE (8slb_AB/AC both use 122_0; 8fbw_AC/BC both use 115_0),
  since one prediction scores both interfaces.

WHY NOT DELETE THE MSA PATHS. msa_search.py:44 treats absent pairedMsaPath/unpairedMsaPath as
"MSA missing" and launches a fresh MSA search -- the opposite of the intended condition. Pointing both
at a 1-sequence a3m is the only reliable way; the run then logs `N_msa 1`.

VERIFICATION per run (a condition difference is invisible in the output):
  `Found 1 templates`  the donor template really loaded
  `N_msa 1`            the MSA really is single-sequence
  no `UNMASKED` line   the mask really is stock
A run failing any of these is reported FAILED and not scored.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/nomsa_donors.py [--gpu 7] [--set rescued]
Writes: artifacts/sweep_tmpl/nomsa_donors/*.json, artifacts/sweep_a3m/nomsa_donors/*.a3m,
        artifacts/nomsa_donors_jobs.json, artifacts/nomsa_donors_logs/*.log
"""
import argparse, glob, json, os, random, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="7")
ap.add_argument("--subset", default="rescued", choices=["rescued", "failed", "all"])
ap.add_argument("--n-seeds", type=int, default=10)
ap.add_argument("--rng", type=int, default=0)
ap.add_argument("--build-only", action="store_true")
ap.add_argument("--only", default="", help="comma list of run names to run "
                    "(e.g. splitting the sweep across machines); default all")
A = ap.parse_args()

ART = f"{EXP}/artifacts"
TMPL = f"{ART}/sweep_tmpl/nomsa_donors"
A3MD = f"{ART}/sweep_a3m/nomsa_donors"
LOGD = f"{ART}/nomsa_donors_logs"
OUTR = f"{ART}/sweep_out/nomsa_donors"
for d in (TMPL, A3MD, LOGD, OUTR):
    os.makedirs(d, exist_ok=True)
PY = "/home/ebrahim/Protenix/.venv/bin/python"

scores = json.load(open(f"{ART}/dockqdonor_scores.json"))
donors = json.load(open(f"{ART}/dockq_arm_donors.json"))
jobs = {j["interface"]: j for j in json.load(open(f"{ART}/dockq_arm_jobs.json"))}

want = [k for k, v in scores.items()
        if A.subset == "all" or (v["rescued"] if A.subset == "rescued" else not v["rescued"])]

# collapse interfaces that share a pdb AND a donor -- one prediction scores both
groups = {}
for iface in sorted(want):
    d = donors.get(iface)
    if not d or iface not in jobs:
        print(f"  SKIP {iface}: no donor/job record"); continue
    key = (d["pdb"], d["seed"], d["sample"])
    groups.setdefault(key, {"ifaces": [], "job": jobs[iface], "donor": d})["ifaces"].append(iface)

built = []
for (pdb, s, k), g in sorted(groups.items()):
    src = g["job"]["json"]
    obj = json.load(open(src)); obj = obj[0] if isinstance(obj, list) else obj
    name = f"{pdb}_nomsa_{s}_{k}"
    n = 0
    for i, e in enumerate(obj["sequences"]):
        if "proteinChain" not in e:
            continue
        p = e["proteinChain"]; q = p["sequence"]
        tp = p.get("templatesPath") or ""
        assert tp and "hmmsearch" not in tp, \
            f"{name} q{i}: expected an INJECTED donor template, got {tp!r}"
        for kind in ("pairedMsaPath", "unpairedMsaPath"):
            fp = f"{A3MD}/{name}_q{i}_{kind}.a3m"
            open(fp, "w").write(f">query\n{q}\n")
            p[kind] = fp
        n += 1
    if A.n_seeds >= 50:
        seeds = list(range(101, 151))          # the full dockqdonor range; own seed included by construction
    else:
        rng = random.Random(A.rng + s * 1000 + k)
        pool = [x for x in range(101, 151) if x != s]
        seeds = [s] + sorted(rng.sample(pool, A.n_seeds - 1))   # own seed ALWAYS first
    assert len(set(seeds)) == A.n_seeds and s in seeds and all(101 <= x <= 150 for x in seeds)
    obj["name"] = name
    jp = f"{TMPL}/{name}.json"
    json.dump([obj], open(jp, "w"), indent=1)
    built.append({"name": name, "pdb": pdb, "json": jp, "ifaces": g["ifaces"],
                  "donor_seed": s, "donor_sample": k,
                  "donor_dockq": g["donor"].get("donor_dockq"),
                  "chains_templated": n, "seeds": seeds})
    print(f"built {name}: {n} chains, covers {g['ifaces']}, seeds {seeds}")

def _tokens(jp):
    try:
        o = json.load(open(jp)); o = o[0] if isinstance(o, list) else o
        return sum(len(e["proteinChain"]["sequence"]) * int(e["proteinChain"].get("count", 1))
                   for e in o["sequences"] if "proteinChain" in e)
    except Exception:
        return 0


# SHORTEST-FIRST: runtime scales steeply with token count (fitted exponent ~2.5 from the two timed
# runs), so 8pn0+8bf4 alone are a third of the total. Ordering by cost banks most interfaces early
# instead of leaving 20 of them behind one giant.
built.sort(key=lambda b: _tokens(b["json"]))
JOBS_F = f"{ART}/nomsa_donors_jobs.json"
if A.only:
    keep = {x.strip() for x in A.only.split(",") if x.strip()}
    missing = keep - {b["name"] for b in built}
    assert not missing, f"--only names not found: {sorted(missing)}"
    built = [b for b in built if b["name"] in keep]
    # write to a SEPARATE file -- overwriting the shared one truncated it to the --only subset
    # and would have made the scorer silently cover 6 of 23 interfaces.
    JOBS_F = f"{ART}/nomsa_donors_jobs_subset.json"
    print(f"--only: restricted to {len(built)} runs -> {JOBS_F} (shared jobs file untouched)")
json.dump(built, open(JOBS_F, "w"), indent=1)
print(f"\n{len(built)} runs covering {sum(len(b['ifaces']) for b in built)} interfaces")
if A.build_only:
    sys.exit(0)


def verify(logf):
    txt = open(logf, errors="ignore").read()
    t = txt.count("Found 1 templates")
    nm = sorted({int(x) for x in __import__("re").findall(r"N_msa (\d+)", txt)})
    unmask = "UNMASKED TEMPLATES ACTIVE" in txt
    probs = []
    if t == 0: probs.append("template never loaded")
    if not nm: probs.append("no N_msa line")
    elif max(nm) > 2: probs.append(f"MSA not single-sequence: N_msa {nm}")
    if unmask: probs.append("ran UNMASKED (should be stock mask)")
    return {"templates": t, "n_msa": nm, "unmasked": unmask, "problems": probs, "ok": not probs}


results = {}
for b in built:
    logf = f"{LOGD}/{b['name']}.log"
    out = f"{OUTR}/{b['name']}"
    os.makedirs(out, exist_ok=True)
    print(f"[{time.strftime('%H:%M')}] {b['name']} (covers {b['ifaces']})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=A.gpu, LAYERNORM_TYPE="openfold",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    # RESUME POISONING GUARD. Seeds with existing output are skipped -- so if a PREVIOUS attempt ran
    # untemplated (the classic `Found 0 templates`, e.g. a donor mmCIF missing from
    # release_date_cache.json), a corrected rerun would treat that garbage as done and keep it. Cost a
    # second machine 49 wasted seeds on 2026-08-10. Refuse to resume over a poisoned run.
    if os.path.exists(logf) and glob.glob(f"{out}/*/seed_*/predictions/*_summary_confidence_sample_4.json"):
        nulls = sum(1 for l in open(logf, errors="ignore") if "Found 0 templates" in l)
        if nulls:
            raise SystemExit(
                "ABORT " + b["name"] + ": its existing log has " + str(nulls) +
                " 'Found 0 templates' lines, so the seeds already on disk are UNTEMPLATED.\n"
                "  Resume would silently keep them. Delete first:\n"
                "    rm -rf " + out + " " + logf)
    todo = [x for x in b["seeds"]
            if not glob.glob(f"{out}/*/seed_{x}/predictions/*_summary_confidence_sample_4.json")]
    if not todo:
        print(f"[{time.strftime('%H:%M')}] {b['name']} already complete, skipping", flush=True)
        continue
    if len(todo) < len(b["seeds"]):
        print(f"    resuming: {len(b['seeds']) - len(todo)} seeds already done, {len(todo)} to run",
              flush=True)
    t0 = time.time()
    with open(logf, "w") as fh:
        subprocess.run([PY, "runner/inference.py", "--model_name", "protenix-v2",
                        "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
                        "--input_json_path", b["json"], "--dump_dir", out,
                        "--seeds", ",".join(map(str, todo)), "--sample_diffusion.N_sample", "5",
                        "--sample_diffusion.N_step", "200", "--use_template", "true"],
                       cwd="/home/ebrahim/Protenix", env=env, stdout=fh, stderr=subprocess.STDOUT)
    v = verify(logf); v["minutes"] = round((time.time() - t0) / 60, 1)
    results[b["name"]] = {**b, **v}
    print(f"[{time.strftime('%H:%M')}] {b['name']} {v['minutes']}min | "
          + ("VERIFIED" if v["ok"] else "FAILED: " + "; ".join(v["problems"]))
          + f" | N_msa={v['n_msa']} templates={v['templates']}", flush=True)
    json.dump(results, open(f"{ART}/nomsa_donors_runs.json", "w"), indent=1)

print("\nALL DONE")
for n, v in results.items():
    print(f"  {n}: " + ("VERIFIED" if v["ok"] else "FAILED " + "; ".join(v["problems"])))
