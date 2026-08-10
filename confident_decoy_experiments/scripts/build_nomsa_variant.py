#!/usr/bin/env python
"""Make a no-MSA twin of an existing allpose arm: same 250 templates, MSA replaced by the query.

Takes <run>_jobs.json (built by build_allpose_generic.py) and writes <run>nomsa_jobs.json where every
chain's pairedMsaPath/unpairedMsaPath points at a 1-sequence a3m containing only the query. The
templatesPath, the injected mmCIF ids, the donors and the seed lists are all UNCHANGED, so the two
arms differ in exactly one thing.

WHY NOT DELETE THE MSA PATHS. msa_search.py:44 treats absent pairedMsaPath/unpairedMsaPath as "MSA
missing" and launches a fresh MSA search -- the opposite of the intended condition. Pointing both at a
1-sequence a3m is the only reliable way; the run then logs `N_msa 1`, which is the check that proves it.

NO NEW TEMPLATE IDS are minted. The mmCIFs and their release_date_cache entries already exist from the
MSA run, so this cannot hit the 4-char-id trap or the missing-release-date trap.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/build_nomsa_variant.py --run allpose8fas10
Writes: artifacts/sweep_tmpl/<run>nomsa/*.json, artifacts/sweep_a3m/<run>nomsa/*.a3m,
        artifacts/<run>nomsa_jobs.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="existing allpose run name, e.g. allpose8fas10")
A = ap.parse_args()

ART = f"{EXP}/artifacts"
SRC = f"{ART}/{A.run}_jobs.json"
NEW = f"{A.run}nomsa"
TMPL = f"{ART}/sweep_tmpl/{NEW}"
A3M = f"{ART}/sweep_a3m/{NEW}"
os.makedirs(TMPL, exist_ok=True); os.makedirs(A3M, exist_ok=True)

jobs = json.load(open(SRC))
out = []
for j in jobs:
    o = json.load(open(j["json"])); o = o[0] if isinstance(o, list) else o
    n = 0
    for i, e in enumerate(o["sequences"]):
        p = e.get("proteinChain")
        if not p:
            continue
        tp = p.get("templatesPath") or ""
        assert tp and "hmmsearch" not in tp, \
            f"{j['name']} q{i}: expected an INJECTED donor template, got {tp!r}"
        assert os.path.exists(tp), f"{j['name']} q{i}: templatesPath missing: {tp}"
        q = p["sequence"]
        for kind in ("pairedMsaPath", "unpairedMsaPath"):
            fp = f"{A3M}/{j['name']}_q{i}_{kind}.a3m"
            open(fp, "w").write(f">query\n{q}\n")
            p[kind] = fp
        n += 1
    name = f"{j['name']}_nomsa"
    o["name"] = name
    jp = f"{TMPL}/{name}.json"
    json.dump([o], open(jp, "w"))
    k = dict(j); k.update({"run": NEW, "name": name, "json": jp, "interface": name,
                           "chains_nomsa": n})
    out.append(k)

json.dump(out, open(f"{ART}/{NEW}_jobs.json", "w"), indent=1)
assert len(out) == len(jobs), "job count changed"
assert len({x["name"] for x in out}) == len(out), "duplicate job name -- outputs would collide"
nch = {x["chains_nomsa"] for x in out}
assert len(nch) == 1, f"jobs rewrote different numbers of chains: {nch}"
tot = sum(len(x["seeds"]) for x in out)
print(f"BUILT {NEW}: {len(out)} donor jobs x {len(out[0]['seeds'])} seeds = {tot} seeds "
      f"({tot * 5} poses), {nch.pop()} chains set to single-sequence")
print(f"   templates reused from {A.run} (same mmCIF ids, no new ids minted)")
print(f"   -> {ART}/{NEW}_jobs.json")
