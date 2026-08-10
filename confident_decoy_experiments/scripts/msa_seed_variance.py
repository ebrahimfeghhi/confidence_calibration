#!/usr/bin/env python
"""Does disabling MSA row subsampling reduce SEED-TO-SEED variance?

Paired comparison over the 15 interfaces of the full-MSA run (fullmsa_n50, seeds 101-150,
--data.msa.min_size.test 16384) against the seed-matched free run (same seeds 101-150,
subsampling ON).

For each interface and each arm we compute, per seed, the best DockQ of its 5 diffusion
samples, then summarise the spread of those 50 per-seed values:
  sd_between   population SD of per-seed best      <- seed-to-seed variance
  iqr_between  p25-p75 of per-seed best
  within       mean (max-min) of the 5 samples in a seed  <- sample-to-sample variance

If MSA subsampling is a major source of seed variance, full MSA should show LOWER
sd_between on most of the 15.

Free per-pose DockQ lives in two caches with two naming schemes (see memory note):
  decoy_set_ost / extra_set_ost : {interface}_{seed}_{sample}.json   (interface-keyed)
  abag_all_ost                  : {pdb}_{seed}_{sample}.json         (pdb-keyed)
Chain names can differ between the fullmsa label and the decoy-set label (e.g. 7z2m_AJ vs
7z2m_AK), so the interface is always matched BY CHAIN PAIR inside the file, never by filename,
and every arm must yield exactly SEEDS x 5 poses or the interface is reported as INCOMPLETE
rather than silently averaged over a partial pool.

Usage: /home/ebrahim/Protenix/.venv/bin/python scripts/msa_seed_variance.py
Writes: artifacts/msa_seed_variance.json
"""
import json, glob, os, sys, statistics as st
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import EXP

A = f"{EXP}/artifacts"
SEEDS = list(range(101, 151))
NSAMP = 5


def dockq_for(path, c1, c2):
    """DockQ of the (c1,c2) interface in one OST json, matched BY CHAIN NAME. None if absent."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            return qs[i]
    return None


def collect(files, seed_of, c1, c2):
    """{seed: [dockq,...]} restricted to SEEDS."""
    per = defaultdict(list)
    for f in files:
        s = seed_of(f)
        if s not in SEEDS:
            continue
        q = dockq_for(f, c1, c2)
        if q is not None:
            per[s].append(q)
    return per


def free_poses(iface, c1, c2):
    """Free-run per-pose DockQ for one interface, trying all three caches. Returns (per, cache)."""
    pdb = iface.split("_")[0]
    for cache, pat, seedfn in (
        ("decoy_set_ost", f"{A}/decoy_set_ost/{iface}_*.json",
         lambda f: int(os.path.basename(f)[:-5].split("_")[-2])),
        ("extra_set_ost", f"{A}/extra_set_ost/{iface}_*.json",
         lambda f: int(os.path.basename(f)[:-5].split("_")[-2])),
        ("abag_all_ost", f"{A}/abag_all_ost/{pdb}_*.json",
         lambda f: int(os.path.basename(f)[:-5].split("_")[-2])),
    ):
        files = glob.glob(pat)
        if not files:
            continue
        per = collect(files, seedfn, c1, c2)
        if sum(len(v) for v in per.values()) >= len(SEEDS) * NSAMP:
            return per, cache
    return None, None


def spread(per):
    best = [max(v) for v in per.values()]
    within = [max(v) - min(v) for v in per.values() if len(v) >= NSAMP]
    b = sorted(best, reverse=True)
    return {
        "n_seeds": len(best), "n_poses": sum(len(v) for v in per.values()),
        "median_best": round(st.median(best), 3),
        "sd_between": round(st.pstdev(best), 4),
        "iqr_between": round(b[int(.25 * len(b))] - b[int(.75 * len(b))], 4),
        "within": round(st.mean(within), 4) if within else None,
    }


ifaces = sorted({os.path.basename(f)[3:].rsplit("_", 2)[0]
                 for f in glob.glob(f"{A}/fullmsa_n50/dq_*.json")
                 if not os.path.basename(f).startswith(("dq_ld_", "dq_nat_"))})
ifaces = [i for i in ifaces if len(i.split("_")) == 2 and len(i.split("_")[1]) == 2]

rows, skipped = [], []
for iface in ifaces:
    c1, c2 = iface.split("_")[1]
    fm = collect(glob.glob(f"{A}/fullmsa_n50/dq_{iface}_*.json"),
                 lambda f: int(os.path.basename(f)[:-5].split("_")[-2]), c1, c2)
    fr, cache = free_poses(iface, c1, c2)
    if fr is None:
        skipped.append((iface, "no free cache with a full pool")); continue
    nfm, nfr = sum(len(v) for v in fm.values()), sum(len(v) for v in fr.values())
    if nfm != len(SEEDS) * NSAMP or nfr < len(SEEDS) * NSAMP:
        skipped.append((iface, f"INCOMPLETE fullmsa={nfm} free={nfr}")); continue
    rows.append({"interface": iface, "free_cache": cache,
                 "fullmsa": spread(fm), "free": spread(fr)})

print(f"paired interfaces: {len(rows)}   skipped: {len(skipped)}")
for i, why in skipped:
    print(f"  SKIP {i}: {why}")

print(f"\n{'interface':>10} {'sd free':>8} {'sd fMSA':>8} {'delta':>7} | "
      f"{'within free':>11} {'within fMSA':>11} | {'med free':>8} {'med fMSA':>8}")
wins = 0
for r in rows:
    a, b = r["free"], r["fullmsa"]
    d = b["sd_between"] - a["sd_between"]
    wins += d < 0
    print(f"{r['interface']:>10} {a['sd_between']:>8.3f} {b['sd_between']:>8.3f} {d:>+7.3f} | "
          f"{a['within']:>11.3f} {b['within']:>11.3f} | {a['median_best']:>8.3f} {b['median_best']:>8.3f}")

sd_f = [r["free"]["sd_between"] for r in rows]
sd_m = [r["fullmsa"]["sd_between"] for r in rows]
wi_f = [r["free"]["within"] for r in rows]
wi_m = [r["fullmsa"]["within"] for r in rows]
md_f = [r["free"]["median_best"] for r in rows]
md_m = [r["fullmsa"]["median_best"] for r in rows]
print(f"\nfull MSA has LOWER seed-to-seed SD on {wins}/{len(rows)} interfaces")
print(f"  mean sd_between   free {st.mean(sd_f):.4f}  fullMSA {st.mean(sd_m):.4f}")
print(f"  mean within-seed  free {st.mean(wi_f):.4f}  fullMSA {st.mean(wi_m):.4f}")
print(f"  mean median best  free {st.mean(md_f):.4f}  fullMSA {st.mean(md_m):.4f}")

json.dump({"seeds": [SEEDS[0], SEEDS[-1]], "n_paired": len(rows),
           "skipped": skipped, "rows": rows},
          open(f"{A}/msa_seed_variance.json", "w"), indent=1)
print(f"\nWROTE {A}/msa_seed_variance.json")
