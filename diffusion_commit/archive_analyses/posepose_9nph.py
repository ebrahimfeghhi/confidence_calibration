#!/usr/bin/env python
"""Pose-pose DockQ over Arm B of 9nph_AC, and a picture of it.

ARM B = the 25 free donor seeds (300000-300024, 5 poses each) + the 25 self-templated runs built from
them (5 poses each) = 250 poses, the same budget as Arm A's 50 free seeds.

WHY. Arm B ships a DockQ 0.064 pose at ipTM 0.824 while a 0.362 pose sits in the same pool at 0.720.
Pose-pose DockQ says whether that 0.064 pose is a lone oddity or the centre of a large basin the model
keeps falling into -- and whether free and templated runs land in the same basins or different ones.

CHAIN MAPPING IS PER POSE, taken from each pose's own PXMeter JSON. The interface is named in
REFERENCE letters (A,C) but the kernel needs MODEL chains, and the two differ per pose. Substituting
one for the other is the mistake that once put 45% of this project's interface measurements on the
wrong chains.

The kernel (scripts/dockq_torch.py) is exact DockQ 2.1.3, vectorised over all pairs; it runs
on CPU here so the GPUs stay untouched.

--hist replots from the saved .npz (no GPU, no re-scoring): the distribution of pose-pose DockQ
between the pose Arm B actually ships (highest ipTM) and every other pose in the pool. It answers
"how much company does the shipped pose keep, and is that company good or bad?"

Usage: posepose_9nph.py [--device cpu] | posepose_9nph.py --hist
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag/confident_decoy_experiments/scripts")
EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
M = f"{EXP}/artifacts/hybrid25"
PDB, C1, C2 = "9nph", "A", "C"
LO, HI = 300000, 300024


def mapping(jf):
    try:
        J = json.load(open(jf))
    except Exception:
        return None, None, None
    r2m = J.get("ref_to_model_chain_mapping", {})
    v = (J.get("interface") or {}).get(f"{C1},{C2}") or (J.get("interface") or {}).get(f"{C2},{C1}")
    if C1 not in r2m or C2 not in r2m or not v or v.get("dockq") is None:
        return None, None, None
    return r2m[C1], r2m[C2], float(v["dockq"])


def central(npz):
    """Does the consensus signal survive templating?

    Centrality = a pose's mean pose-pose DockQ to the other poses in its own pool. Measured over all
    250 Arm B poses the good basin was the more central one, which is the opposite of what ipTM says
    and therefore a candidate label-free selector. But templating AMPLIFIES whichever basin it is fed,
    so it can inflate the decoy basin's population and tightness. If centrality only separates the
    basins in the free pool and collapses in the templated pool, it is useless exactly where we would
    want to apply it -- on the method's own output.

    Pools are compared WITHIN themselves (free vs free, templated vs templated) because centrality is
    relative to whatever pool a selector would actually see.
    """
    ACC = 0.23
    Z = np.load(npz, allow_pickle=True)
    D = Z["M"].astype(np.float64)
    S = (D + D.T) / 2.0
    meta = [json.loads(s) for s in Z["meta"]]
    arm = np.array([m["arm"] for m in meta])
    dq = np.array([m["dockq"] for m in meta])
    ip = np.array([m["iptm"] for m in meta])

    print(f"{'pool':<22} {'basin':<18} {'n':>4} {'centrality':>11} {'mean ipTM':>10}")
    for pool, m in (("free (125)", arm == "free"), ("templated (125)", arm == "tmpl"),
                    ("both (250)", np.ones(len(meta), bool))):
        idx = np.where(m)[0]
        sub = S[np.ix_(idx, idx)].copy()
        np.fill_diagonal(sub, np.nan)
        cen = np.nanmean(sub, axis=1)
        gap = []
        for lab, keep in (("acceptable", True), ("decoy", False)):
            sel = (dq[idx] >= ACC) == keep
            if not sel.any():
                print(f"{pool:<22} {lab:<18} {0:>4}")
                continue
            gap.append(cen[sel].mean())
            print(f"{pool:<22} {lab:<18} {int(sel.sum()):>4} {cen[sel].mean():>11.3f} "
                  f"{np.nanmean(ip[idx][sel]):>10.3f}")
        if len(gap) == 2:
            # would argmax-centrality ship an acceptable pose from this pool?
            top = idx[int(np.nanargmax(cen))]
            tip = idx[int(np.nanargmax(ip[idx]))]
            print(f"{'':<22} {'-> gap (acc-decoy)':<18} {'':>4} {gap[0]-gap[1]:>+11.3f}")
            print(f"{'':<22} most CENTRAL pose: DockQ {dq[top]:.3f} "
                  f"({'ACCEPTABLE' if dq[top] >= ACC else 'decoy'})   "
                  f"most CONFIDENT pose: DockQ {dq[tip]:.3f} "
                  f"({'ACCEPTABLE' if dq[tip] >= ACC else 'decoy'})")


def spread(png):
    """Within-seed DockQ spread across the 5 diffusion samples, over all Arm A (free) seeds.

    ARM A is the 50-seed free sweep -- the baseline both arms are compared against. The question is
    how much of a seed's outcome is decided before diffusion: a seed with near-zero spread has
    committed to one docking basin in the trunk and the five samples only jitter within it, whereas a
    wide seed is still choosing. On 85 selection failures elsewhere in this project the confident seed
    was frozen (40/85 had exactly zero spread), so the shape of this distribution is the question of
    whether diffusion is a real search or a rounding error.

    Bars are split by whether that seed produced any acceptable pose, since a wide seed that never
    crosses 0.23 means something different from one that straddles it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ACC = 0.23
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if (r["pdb"] != PDB or not r["dockq"]
                or "".join(sorted((r["chain1"], r["chain2"]))) != "".join(sorted((C1, C2)))):
            continue
        by[int(r["seed"])].append(float(r["dockq"]))

    seeds = sorted(by)
    rng = np.array([max(by[s]) - min(by[s]) for s in seeds])
    sd = np.array([np.std(by[s], ddof=0) for s in seeds])
    nsamp = np.array([len(by[s]) for s in seeds])
    hit = np.array([max(by[s]) >= ACC for s in seeds])
    straddle = np.array([max(by[s]) >= ACC > min(by[s]) for s in seeds])

    print(f"Arm A seeds: {len(seeds)}   samples per seed: "
          f"{dict(collections.Counter(nsamp.tolist()))}")
    print(f"within-seed DockQ RANGE (max-min): median {np.median(rng):.3f}  mean {rng.mean():.3f}  "
          f"max {rng.max():.3f}")
    print(f"   range < 0.05 (effectively frozen): {int((rng < 0.05).sum())}/{len(seeds)}")
    print(f"   exactly 0.000:                     {int((rng == 0).sum())}/{len(seeds)}")
    print(f"within-seed DockQ SD: median {np.median(sd):.3f}  mean {sd.mean():.3f}")
    print(f"seeds with any acceptable pose: {int(hit.sum())}/{len(seeds)}   "
          f"straddling 0.23 (some good, some not): {int(straddle.sum())}/{len(seeds)}")
    print(f"   mean range | seed has a good pose: {rng[hit].mean():.3f} (n={int(hit.sum())})")
    print(f"   mean range | seed has none:        {rng[~hit].mean():.3f} (n={int((~hit).sum())})")

    # Does the seed ipTM ships have more or less spread than the seeds that hold good poses? On 85
    # selection failures elsewhere in this project the confident seed was FROZEN, so the prediction
    # is less spread, not more.
    ipby = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if (r["pdb"] == PDB and r["pair_iptm"]
                and "".join(sorted((r["chain1"], r["chain2"]))) == "".join(sorted((C1, C2)))):
            ipby[int(r["seed"])].append(float(r["pair_iptm"]))
    mip = np.array([np.mean(ipby[s]) if ipby.get(s) else np.nan for s in seeds])
    xip = np.array([max(ipby[s]) if ipby.get(s) else np.nan for s in seeds])
    ship = int(np.nanargmax(xip))
    print(f"\nseed ipTM ships (highest max pair_iptm): seed {seeds[ship]}  "
          f"max ipTM {xip[ship]:.3f}  best DockQ {max(by[seeds[ship]]):.3f}  "
          f"range {rng[ship]:.3f}  acceptable: {bool(hit[ship])}")
    for lab, m in (("seeds WITH an acceptable pose", hit), ("seeds with NONE", ~hit)):
        print(f"   {lab:<30} n={int(m.sum()):>3}  mean range {rng[m].mean():.3f}  "
              f"mean max-ipTM {np.nanmean(xip[m]):.3f}  mean best-DockQ "
              f"{np.mean([max(by[seeds[i]]) for i in range(len(seeds)) if m[i]]):.3f}")
    o = np.argsort(-xip)[:8]
    print(f"\n   top 8 Arm A seeds by max pair_iptm")
    print(f"   {'seed':>6} {'max ipTM':>9} {'best DockQ':>11} {'range':>7} {'acceptable':>11}")
    for i in o:
        print(f"   {seeds[i]:>6} {xip[i]:>9.3f} {max(by[seeds[i]]):>11.3f} {rng[i]:>7.3f} "
              f"{str(bool(hit[i])):>11}")

    # ARM B, the templated half: is the run ipTM ships MORE scattered than the most confident
    # acceptable run? If the shipped decoy is tight, confidence is still tracking commitment and
    # templating merely moved which basin got committed to. If it is scattered, commitment is not
    # the story.
    tb = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{M}/results_all.csv")):
        if r["interface"] != f"{PDB}_{''.join(sorted((C1, C2)))}":
            continue
        try:
            tb[int(r["donor_seed"])].append((float(r["dockq"]), float(r["iptm"])))
        except ValueError:
            continue
    tseeds = sorted(tb)
    print(f"\nARM B templated runs: {len(tseeds)} (samples per run: "
          f"{dict(collections.Counter(len(tb[s]) for s in tseeds))})")
    print(f"   {'run':>8} {'max ipTM':>9} {'DockQ rng':>10} {'ipTM rng':>9} {'best DockQ':>11} "
          f"{'good/5':>7}")
    rows = []
    for s in tseeds:
        d = [x[0] for x in tb[s]]; i = [x[1] for x in tb[s]]
        rows.append((s, max(i), max(d) - min(d), max(i) - min(i), max(d),
                     sum(1 for x in d if x >= ACC)))
    for r in sorted(rows, key=lambda r: -r[1])[:8]:
        print(f"   tmpl{r[0]:>4} {r[1]:>9.3f} {r[2]:>10.3f} {r[3]:>9.3f} {r[4]:>11.3f} {r[5]:>7}")
    acc_rows = [r for r in rows if r[5] > 0]
    if acc_rows:
        top_acc = max(acc_rows, key=lambda r: r[1])
        top_any = max(rows, key=lambda r: r[1])
        print(f"   shipped run      tmpl{top_any[0]}: DockQ range {top_any[2]:.3f}, "
              f"ipTM range {top_any[3]:.3f}, good {top_any[5]}/5")
        print(f"   best ACCEPTABLE  tmpl{top_acc[0]}: DockQ range {top_acc[2]:.3f}, "
              f"ipTM range {top_acc[3]:.3f}, good {top_acc[5]}/5")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bins = np.linspace(0, max(0.05, rng.max() * 1.05), 26)
    ax.hist([rng[~hit], rng[hit]], bins=bins, stacked=True,
            color=["#b03a48", "#2f7fb8"], edgecolor="white", linewidth=.4,
            label=[f"seed never reaches 0.23 (n={int((~hit).sum())})",
                   f"seed has >=1 acceptable pose (n={int(hit.sum())})"])
    ax.axvline(np.median(rng), color="0.25", ls="--", lw=1.2,
               label=f"median range {np.median(rng):.3f}")
    ax.set_xlabel("within-seed DockQ range across the 5 diffusion samples (max - min)")
    ax.set_ylabel("number of seeds")
    ax.set_title(f"9nph_AC Arm A: how much does diffusion move a seed?\n"
                 f"{len(seeds)} free seeds x 5 samples")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


def lift():
    """Is self-templating a BASIN-SYMMETRIC confidence amplifier, or does it favour the decoy?

    Two readings, because they answer different questions:

    (1) POPULATION 2x2 -- mean ipTM by arm (free / templated) x basin (DockQ >= 0.23 or not). Shows
        where confidence ends up, but the two arms need not contain the same mix of poses, so a gap
        here can be composition rather than lift.

    (2) PAIRED BY DONOR -- the unit hybrid25 actually ran. Donor seed s contributes ALL FIVE of its
        poses as templates for run seed s+25, so the donor is the seed, not the pose. Lift is
        mean ipTM of the five outputs minus mean ipTM of the five donors. Donors are then grouped by
        how many of their five poses were acceptable (5/5 good, mixed, 0/5 good), because a templated
        run inherits its donor's basin. An interaction -- larger lift for decoy donors -- is the
        claim under test.

    The donor and output share a basin only in expectation, so output basin is reported too.
    """
    ACC = 0.23
    free = collections.defaultdict(dict)      # seed -> sample -> (dockq, iptm)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if (r["pdb"] != PDB or not r["dockq"] or not r["pair_iptm"]
                or "".join(sorted((r["chain1"], r["chain2"]))) != "".join(sorted((C1, C2)))):
            continue
        s = int(r["seed"])
        if LO <= s <= HI:
            free[s][int(r["sample"])] = (float(r["dockq"]), float(r["pair_iptm"]))

    tm = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{M}/results_all.csv")):
        if r["interface"] != f"{PDB}_{''.join(sorted((C1, C2)))}":
            continue
        try:
            tm[int(r["donor_seed"])][int(r["sample"])] = (float(r["dockq"]), float(r["iptm"]))
        except ValueError:
            continue

    # (1) population 2x2
    print(f"\n(1) POPULATION 2x2 -- mean ipTM, cells are poses\n")
    print(f"{'basin':<22} {'free ipTM':>16} {'templated ipTM':>18} {'delta':>8}")
    for lab, keep in (("acceptable (>=0.23)", True), ("decoy (<0.23)", False)):
        fv = [ip for s in free for d, ip in [free[s][k] for k in free[s]] if (d >= ACC) == keep]
        tv = [ip for s in tm for d, ip in [tm[s][k] for k in tm[s]] if (d >= ACC) == keep]
        print(f"{lab:<22} {np.mean(fv):>9.3f} (n={len(fv):>3}) "
              f"{np.mean(tv):>11.3f} (n={len(tv):>3}) {np.mean(tv)-np.mean(fv):>+8.3f}")

    # (2) paired by donor seed
    print(f"\n(2) PAIRED BY DONOR SEED -- five donor poses -> five templated poses\n")
    rows = []
    for s in sorted(set(free) & set(tm)):
        dv = list(free[s].values()); ov = list(tm[s].values())
        if not dv or not ov:
            continue
        ngood = sum(1 for d, _ in dv if d >= ACC)
        rows.append(dict(seed=s, ngood=ngood, n_d=len(dv), n_o=len(ov),
                         dip=float(np.mean([ip for _, ip in dv])),
                         oip=float(np.mean([ip for _, ip in ov])),
                         ddq=float(np.mean([d for d, _ in dv])),
                         odq=float(np.mean([d for d, _ in ov])),
                         ogood=sum(1 for d, _ in ov if d >= ACC)))
    print(f"{'donor basin':<26} {'donors':>7} {'donor ipTM':>11} {'out ipTM':>10} {'LIFT':>8} "
          f"{'donor DockQ':>12} {'out DockQ':>10}")
    groups = (("all 5 donor poses good", lambda r: r["ngood"] == 5),
              ("mixed (1-4 good)",       lambda r: 1 <= r["ngood"] <= 4),
              ("all 5 donor poses decoy", lambda r: r["ngood"] == 0))
    for lab, f in groups:
        v = [r for r in rows if f(r)]
        if not v:
            print(f"{lab:<26} {0:>7}")
            continue
        li = np.array([r["oip"] - r["dip"] for r in v])
        print(f"{lab:<26} {len(v):>7} {np.mean([r['dip'] for r in v]):>11.3f} "
              f"{np.mean([r['oip'] for r in v]):>10.3f} {li.mean():>+8.3f} "
              f"{np.mean([r['ddq'] for r in v]):>12.3f} {np.mean([r['odq'] for r in v]):>10.3f}")
    li_all = np.array([r["oip"] - r["dip"] for r in rows])
    print(f"{'ALL':<26} {len(rows):>7} {np.mean([r['dip'] for r in rows]):>11.3f} "
          f"{np.mean([r['oip'] for r in rows]):>10.3f} {li_all.mean():>+8.3f} "
          f"{np.mean([r['ddq'] for r in rows]):>12.3f} {np.mean([r['odq'] for r in rows]):>10.3f}")

    print(f"\nper donor seed (n_d/n_o = donor/output poses actually present)")
    print(f"{'seed':>6} {'good/5':>7} {'n_d':>4} {'n_o':>4} {'donor ipTM':>11} {'out ipTM':>9} "
          f"{'lift':>7} {'donor DockQ':>12} {'out DockQ':>10} {'out good/5':>11}")
    for r in sorted(rows, key=lambda r: r["oip"] - r["dip"], reverse=True):
        print(f"{r['seed']:>6} {r['ngood']:>7} {r['n_d']:>4} {r['n_o']:>4} {r['dip']:>11.3f} "
              f"{r['oip']:>9.3f} {r['oip']-r['dip']:>+7.3f} {r['ddq']:>12.3f} {r['odq']:>10.3f} "
              f"{r['ogood']:>11}")


def hist(npz, png):
    """Similarity of the SHIPPED pose (highest ipTM in Arm B) to every other pose in the pool.

    Colour splits each bar by whether the neighbour is itself acceptable (DockQ >= 0.23), so the
    picture says not just how much company the shipped pose keeps but what KIND -- a decoy sitting in
    a crowd of fellow decoys looks very different from one sitting near good poses.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = np.load(npz, allow_pickle=True)
    D = Z["M"].astype(np.float64)
    S = (D + D.T) / 2.0
    meta = [json.loads(s) for s in Z["meta"]]
    win = int(np.nanargmax([m["iptm"] for m in meta]))
    w = meta[win]
    oth = np.array([i for i in range(len(meta)) if i != win])
    sim = S[win][oth]
    good = np.array([meta[i]["dockq"] >= 0.23 for i in oth])

    print(f"shipped: {w['arm']} seed {w['seed']} sample {w['samp']}  "
          f"DockQ {w['dockq']:.3f}  ipTM {w['iptm']:.3f}")
    print(f"n others {len(oth)}   median sim {np.median(sim):.3f}   max sim {sim.max():.3f}")
    print(f"   acceptable neighbours (DockQ>=0.23): {int(good.sum())} of {len(oth)}")
    print(f"   of the 20 MOST similar poses, acceptable: "
          f"{int(good[np.argsort(-sim)[:20]].sum())}/20")

    # the pose ipTM SHOULD have shipped: highest true DockQ in the same pool. If its similarity to the
    # shipped pose is ~0 the two are different structures and no amount of refinement connects them.
    bi = int(np.argmax([m["dockq"] for m in meta]))
    b = meta[bi]
    print(f"best in pool: {b['arm']} seed {b['seed']} sample {b['samp']}  "
          f"DockQ {b['dockq']:.3f}  ipTM {b['iptm']:.3f}")
    print(f"   similarity(shipped, best) = {S[win][bi]:.3f}   "
          f"ipTM rank of best: {int((np.array([m['iptm'] for m in meta]) > b['iptm']).sum()) + 1}"
          f" of {len(meta)}")
    accs = [i for i in range(len(meta)) if meta[i]["dockq"] >= 0.23]
    print(f"   similarity(shipped, each of the {len(accs)} acceptable poses): "
          f"max {S[win][accs].max():.3f}  median {np.median(S[win][accs]):.3f}")

    # The pose a top-k selector would actually have to reach: the MOST CONFIDENT acceptable pose, not
    # the best one. Its ipTM rank is the k that top-k selection would need on this interface.
    ci = max(accs, key=lambda i: meta[i]["iptm"])
    c = meta[ci]
    ips_all = np.array([m["iptm"] for m in meta])
    print(f"most confident ACCEPTABLE pose: {c['arm']} seed {c['seed']} sample {c['samp']}  "
          f"DockQ {c['dockq']:.3f}  ipTM {c['iptm']:.3f}")
    print(f"   similarity(shipped, it) = {S[win][ci]:.3f}   "
          f"ipTM rank {int((ips_all > c['iptm']).sum()) + 1} of {len(meta)}   "
          f"ipTM gap below shipped: {w['iptm'] - c['iptm']:+.3f}")

    # BAR HEIGHT = how many poses fall in that similarity bin; BAR COLOUR = their mean ipTM.
    # Colouring by the mean rather than stacking keeps the height readable as a plain count, and the
    # question here is "are the poses that look like the shipped one also the confident ones?" --
    # which is a property of the bin, not of individual poses.
    ip = np.array([meta[i]["iptm"] for i in oth], dtype=float)
    bins = np.linspace(0, 1, 41)
    idx = np.clip(np.digitize(sim, bins) - 1, 0, len(bins) - 2)
    cnt = np.zeros(len(bins) - 1, dtype=int)
    mip = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        m = idx == b
        cnt[b] = int(m.sum())
        if m.any() and np.isfinite(ip[m]).any():
            mip[b] = np.nanmean(ip[m])

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=np.nanmin(ip), vmax=np.nanmax(ip))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(bins[:-1], cnt, width=np.diff(bins), align="edge",
           color=[cmap(norm(v)) if np.isfinite(v) else "0.85" for v in mip],
           edgecolor="white", linewidth=.4)
    # log count: one bin holds 144 of 249 poses and linearly it erases the tail, which is the part
    # carrying the signal (the decoy basin out at 0.4-0.85).
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(cnt) * 1.8)
    ax.axvline(np.median(sim), color="0.25", ls="--", lw=1.2)
    ax.text(np.median(sim), max(cnt) * 1.5, f" median {np.median(sim):.2f}",
            fontsize=9, color="0.25", va="top")
    # the shipped pose's own confidence, for reference against the bar colours
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.046, label="mean ipTM of poses in bin"
                 ).ax.axhline(w["iptm"], color="red", lw=1.6)
    ax.set_xlabel(f"pose-pose DockQ to the shipped pose "
                  f"(seed {w['seed']} s{w['samp']}, true DockQ {w['dockq']:.3f}, ipTM {w['iptm']:.3f})")
    ax.set_ylabel("number of poses")
    ax.set_title("9nph_AC Arm B: what the most confident pose resembles, coloured by confidence\n"
                 "(red line on the colourbar = the shipped pose's own ipTM)")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")
    print("   bin  n   mean ipTM   mean true DockQ")
    dqo = np.array([meta[i]["dockq"] for i in oth])
    for b in range(len(bins) - 1):
        if cnt[b]:
            m = idx == b
            print(f"   {bins[b]:.2f}-{bins[b+1]:.2f} {cnt[b]:>3}   "
                  f"{mip[b]:.3f}      {np.mean(dqo[m]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=f"{EXP}/artifacts/posepose_9nph.npz")
    ap.add_argument("--png", default=f"{EXP}/artifacts/posepose_9nph.png")
    ap.add_argument("--hist", action="store_true",
                    help="replot the shipped pose's similarity histogram from the saved npz")
    ap.add_argument("--central", action="store_true",
                    help="does the centrality signal survive templating? free vs templated pool")
    ap.add_argument("--spread", action="store_true",
                    help="histogram of within-seed DockQ spread across the 5 samples, Arm A")
    ap.add_argument("--lift", action="store_true",
                    help="ipTM lift from templating, by arm x basin and paired by donor seed")
    A = ap.parse_args()

    if A.hist:
        return hist(A.out, f"{EXP}/artifacts/posepose_9nph_hist.png")
    if A.central:
        return central(A.out)
    if A.spread:
        return spread(f"{EXP}/artifacts/spread_9nph_armA.png")
    if A.lift:
        return lift()

    # ipTM per pose, to colour the picture
    ip_free = {}
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if (r["pdb"] == PDB and r["pair_iptm"]
                and "".join(sorted((r["chain1"], r["chain2"]))) == "".join(sorted((C1, C2)))):
            ip_free[(int(r["seed"]), int(r["sample"]))] = float(r["pair_iptm"])
    ip_tmpl = {}
    for r in csv.DictReader(open(f"{M}/results_all.csv")):
        if r["interface"] == f"{PDB}_{''.join(sorted((C1,C2)))}":
            try:
                ip_tmpl[(int(r["donor_seed"]), int(r["sample"]))] = float(r["iptm"])
            except ValueError:
                pass

    items, meta = [], []
    # free half
    for s in range(LO, HI + 1):
        for k in range(5):
            cif = f"{P}/out/{PDB}/{PDB}/seed_{s}/predictions/{PDB}_sample_{k}.cif"
            m1, m2, dq = mapping(f"{P}/pxm_json/{PDB}_{s}_{k}.json")
            if not os.path.exists(cif) or m1 is None:
                continue
            items.append((cif, m1, m2))
            meta.append(dict(arm="free", seed=s, samp=k, dockq=dq,
                             iptm=ip_free.get((s, k), np.nan)))
    # templated half
    for s in range(LO, HI + 1):
        for k in range(5):
            cifs = glob.glob(f"{M}/{PDB}/s{s}/out/**/*_sample_{k}.cif", recursive=True)
            m1, m2, dq = mapping(f"{M}/{PDB}/s{s}/pxm_json/{k}.json")
            if not cifs or m1 is None:
                continue
            items.append((cifs[0], m1, m2))
            meta.append(dict(arm="tmpl", seed=s, samp=k, dockq=dq,
                             iptm=ip_tmpl.get((s, k), np.nan)))

    print(f"poses: {len(items)}  free={sum(1 for m in meta if m['arm']=='free')}  "
          f"templated={sum(1 for m in meta if m['arm']=='tmpl')}")
    if len(items) < 20:
        sys.exit("too few poses")

    from dockq_torch import dockq_matrix
    D = dockq_matrix(items, device=A.device, verbose=True)
    D = np.asarray(D.cpu() if hasattr(D, "cpu") else D, dtype=np.float64)
    S = (D + D.T) / 2.0
    np.savez_compressed(A.out, M=D, meta=np.array([json.dumps(m) for m in meta]))
    print(f"wrote {A.out}")

    # --- the shipped pose: highest ipTM in Arm B
    ips = np.array([m["iptm"] for m in meta])
    win = int(np.nanargmax(ips))
    w = meta[win]
    print(f"\nArm B ships: {w['arm']} seed {w['seed']} sample {w['samp']}  "
          f"DockQ {w['dockq']:.3f}  ipTM {w['iptm']:.3f}")
    for thr in (0.9, 0.8, 0.7, 0.5):
        nb = np.where(S[win] >= thr)[0]
        nb = nb[nb != win]
        af = sum(1 for i in nb if meta[i]["arm"] == "free")
        print(f"   neighbours within DockQ {thr}: {len(nb):>3}  "
              f"({af} free, {len(nb)-af} templated)   "
              f"their true DockQ: {np.mean([meta[i]['dockq'] for i in nb]):.3f}" if len(nb)
              else f"   neighbours within DockQ {thr}: 0")

    # --- picture: order by arm, then by true DockQ inside each arm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(range(len(meta)), key=lambda i: (meta[i]["arm"] != "free", meta[i]["dockq"]))
    Sp = S[np.ix_(order, order)]
    dq = np.array([meta[i]["dockq"] for i in order])
    arm = np.array([meta[i]["arm"] for i in order])
    nfree = int((arm == "free").sum())
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6),
                           gridspec_kw={"width_ratios": [1, 1]})
    im = ax[0].imshow(Sp, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax[0].axhline(nfree - .5, color="cyan", lw=1.1); ax[0].axvline(nfree - .5, color="cyan", lw=1.1)
    ax[0].set_title(f"9nph_AC Arm B pose-pose DockQ\n(left/top block = 125 free, "
                    f"right/bottom = 125 templated; sorted by true DockQ within each)")
    ax[0].set_xlabel("pose"); ax[0].set_ylabel("pose")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="pose-pose DockQ")
    wpos = order.index(win)
    ax[0].plot(wpos, wpos, marker="s", ms=9, mfc="none", mec="lime", mew=2)
    sc = ax[1].scatter(dq, [np.nanmean(S[order[i]][np.array(order) != order[i]]) for i in range(len(order))],
                       c=[meta[i]["iptm"] for i in order], cmap="viridis", s=26,
                       edgecolor="white", linewidth=.3)
    ax[1].axvline(0.23, color="0.5", ls="--", lw=1)
    ax[1].set_xlabel("true DockQ vs native"); ax[1].set_ylabel("mean pose-pose DockQ to all others")
    ax[1].set_title("how central is each pose, and how confident\n(dashed = acceptability 0.23)")
    fig.colorbar(sc, ax=ax[1], fraction=0.046, label="ipTM")
    fig.tight_layout()
    fig.savefig(A.png, dpi=150)
    print(f"wrote {A.png}")


if __name__ == "__main__":
    main()
