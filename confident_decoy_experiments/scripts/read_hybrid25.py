#!/usr/bin/env python
"""Arm A (50 free seeds) vs Arm B (25 free + 25 self-templated) at MATCHED compute.

Both arms are 250 poses per interface. Arm A is the original PXM-2025 sweep. Arm B replaces the
second 25 seeds with self-templated runs: for each free seed i in 300000-300024, all five of its
poses become templates for a run under seed i+25.

THE METRIC IS WHAT ipTM ACTUALLY SHIPS, not the oracle. On this benchmark ipTM selection is the
failure mode under study -- on 9x1h it ranked the first acceptable pose 159th of 250 -- so the
question is whether spending half the budget refining beats spending all of it exploring, as judged
by the pose a user would actually get.

ONLY COMPLETE INTERFACES ARE COMPARED. A partially-finished interface has fewer templated seeds and
would understate Arm B; those are listed as pending rather than averaged in.

Classes: decoy = ipTM ships a bad pose though a good one exists; nocov = no acceptable pose anywhere
in 50 free seeds (a control -- there is nothing to amplify, so Arm B should not help); solved = ipTM
already ships an acceptable pose (a control for harm).

Usage: read_hybrid25.py
"""
import collections
import csv
import os

import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = f"{EXP}/artifacts/pxm2025"
M = f"{EXP}/artifacts/hybrid25"
ACC = 0.23
FREE_LO, FREE_HI = 300000, 300024          # donor half, shared by both arms


def main():
    # EVERY ab-ag interface, not one per complex: results_all.csv is regenerated from the PXMeter
    # JSONs, which already score all interfaces of each predicted assembly. PXM-2025 has 89
    # confident-decoy interfaces across only 45 complexes, so one-per-complex discarded half the
    # benchmark. Interfaces are the unit the benchmark is defined on.
    src = f"{M}/results_all.csv" if os.path.exists(f"{M}/results_all.csv") else f"{M}/results.csv"
    want = {}
    for r in csv.DictReader(open(src)):
        want.setdefault(r["interface"], {"pdb": r["pdb"], "cls": r["cls"], "n": 0})["n"] += 1

    # free poses, per interface, from the sweep
    free = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        key = f"{r['pdb']}_{''.join(sorted((r['chain1'], r['chain2'])))}"
        if key in want:
            free[key].append((int(r["seed"]), float(r["dockq"]), float(r["pair_iptm"])))

    # templated poses
    tmpl = collections.defaultdict(list)
    seeds_done = collections.defaultdict(set)
    if os.path.exists(src):
        for r in csv.DictReader(open(src)):
            try:
                d, p = float(r["dockq"]), float(r["iptm"])
            except ValueError:
                continue
            tmpl[r["interface"]].append((int(r["donor_seed"]), d, p))
            seeds_done[r["interface"]].add(int(r["donor_seed"]))

    rows, pending = [], collections.Counter()
    for iface, meta in want.items():
        nd = len(seeds_done.get(iface, ()))
        if nd < 25 or iface not in free:
            pending[meta["cls"]] += 1
            continue
        A = free[iface]
        B = [x for x in A if FREE_LO <= x[0] <= FREE_HI] + list(tmpl[iface])
        if not A or not B:
            continue

        def sel(v):                       # what ipTM ships, and the ceiling
            best = max(v, key=lambda x: x[2])
            return best[1], max(x[1] for x in v)
        sa, oa = sel(A)
        sb, ob = sel(B)
        rows.append(dict(iface=iface, cls=meta["cls"], selA=sa, orcA=oa, selB=sb, orcB=ob,
                         nA=len(A), nB=len(B)))

    print(f"\ncomplete interfaces: {len(rows)}   pending: {dict(pending)}")
    if not rows:
        return
    print(f"\n{'class':<8} {'n':>3} {'selected A':>11} {'selected B':>11} {'delta':>7} "
          f"{'solved A':>9} {'solved B':>9} {'oracle A':>9} {'oracle B':>9}")
    for cl in ("decoy", "nocov", "solved", "ALL"):
        v = rows if cl == "ALL" else [r for r in rows if r["cls"] == cl]
        if not v:
            continue
        sa = np.array([r["selA"] for r in v]); sb = np.array([r["selB"] for r in v])
        oa = np.array([r["orcA"] for r in v]); ob = np.array([r["orcB"] for r in v])
        print(f"{cl:<8} {len(v):>3} {sa.mean():>11.3f} {sb.mean():>11.3f} {sb.mean()-sa.mean():>+7.3f} "
              f"{str(int((sa>=ACC).sum()))+'/'+str(len(v)):>9} {str(int((sb>=ACC).sum()))+'/'+str(len(v)):>9} "
              f"{oa.mean():>9.3f} {ob.mean():>9.3f}")
    print(f"\nper-interface (selected DockQ):")
    print(f"{'interface':<12} {'class':<7} {'A':>7} {'B':>7} {'delta':>7}")
    for r in sorted(rows, key=lambda r: r["selB"] - r["selA"], reverse=True):
        print(f"{r['iface']:<12} {r['cls']:<7} {r['selA']:>7.3f} {r['selB']:>7.3f} "
              f"{r['selB']-r['selA']:>+7.3f}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
def fairness():
    """Decompose Arm B's effect into TEMPLATING versus THE 25 SEEDS IT GAVE UP.

    THE CONFOUND. Arm B = free seeds 300000-300024 + 25 templated runs built from them. It never
    contains free seeds 300025-300049. So whenever Arm A ships a good pose that came from a LATE seed,
    Arm B loses it for a reason that has nothing to do with templating -- the pose simply is not in its
    pool. The honest baseline for "what did templating add" is therefore Arm A restricted to its first
    25 seeds (A25, 125 poses), not the full A50.

    Three arms, all from files already on disk:
      A50  50 free seeds, 250 poses   -- the published baseline
      A25  25 free seeds, 125 poses   -- what Arm B actually started from (HALF the compute)
      B    A25 free + 25 templated, 250 poses

    A50 -> B mixes two changes. A25 -> B isolates templating at matched donor coverage; A25 -> A50
    prices the 25 discarded seeds.

    Classes are recomputed per arm rather than inherited, and interfaces missing from the protocol file
    are classified by the same rule from free-arm data instead of being dropped.
    """
    src = f"{M}/results_all.csv" if os.path.exists(f"{M}/results_all.csv") else f"{M}/results.csv"
    want = {}
    for r in csv.DictReader(open(src)):
        want.setdefault(r["interface"], r["pdb"])

    free = collections.defaultdict(list)          # iface -> [(seed, dockq, iptm)]
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        k = f"{r['pdb']}_{''.join(sorted((r['chain1'], r['chain2'])))}"
        if k in want:
            free[k].append((int(r["seed"]), float(r["dockq"]), float(r["pair_iptm"])))
    tmpl = collections.defaultdict(list)
    seeds_done = collections.defaultdict(set)
    for r in csv.DictReader(open(src)):
        try:
            tmpl[r["interface"]].append((int(r["donor_seed"]), float(r["dockq"]), float(r["iptm"])))
            seeds_done[r["interface"]].add(int(r["donor_seed"]))
        except ValueError:
            continue

    def sel(v):
        return max(v, key=lambda x: x[2])[1]

    rows = []
    for iface, pdb in want.items():
        if len(seeds_done.get(iface, ())) < 25 or iface not in free:
            continue
        A50 = free[iface]
        A25 = [x for x in A50 if FREE_LO <= x[0] <= FREE_HI]
        B = A25 + list(tmpl[iface])
        if not A25 or not B:
            continue
        cls = lambda v: ("decoy" if sel(v) < ACC <= max(x[1] for x in v)
                         else ("nocov" if max(x[1] for x in v) < ACC else "solved"))
        # where does A50's SHIPPED pose come from -- an early seed Arm B also had, or a late one?
        shipped_seed = max(A50, key=lambda x: x[2])[0]
        acc_early = any(x[1] >= ACC for x in A25)
        acc_late = any(x[1] >= ACC for x in A50 if x[0] > FREE_HI)
        rows.append(dict(iface=iface, pdb=pdb, cls50=cls(A50), cls25=cls(A25),
                         s50=sel(A50), s25=sel(A25), sB=sel(B),
                         o50=max(x[1] for x in A50), o25=max(x[1] for x in A25),
                         oB=max(x[1] for x in B), shipped_late=shipped_seed > FREE_HI,
                         acc_early=acc_early, acc_late=acc_late))

    print(f"\ninterfaces: {len(rows)}   complexes: {len({r['pdb'] for r in rows})}")

    print(f"\n(1) THREE ARMS, classed by A50   [selected DockQ; solved = shipped pose >= {ACC}]")
    print(f"{'class':<8} {'n':>4} {'A50':>7} {'A25':>7} {'B':>7} {'B-A50':>7} {'B-A25':>7} "
          f"{'solvA50':>8} {'solvA25':>8} {'solvB':>7}")
    for cl in ("decoy", "nocov", "solved", "ALL"):
        v = rows if cl == "ALL" else [r for r in rows if r["cls50"] == cl]
        if not v:
            continue
        f = lambda k: np.mean([r[k] for r in v])
        n = lambda k: sum(1 for r in v if r[k] >= ACC)
        print(f"{cl:<8} {len(v):>4} {f('s50'):>7.3f} {f('s25'):>7.3f} {f('sB'):>7.3f} "
              f"{f('sB')-f('s50'):>+7.3f} {f('sB')-f('s25'):>+7.3f} "
              f"{n('s50'):>8} {n('s25'):>8} {n('sB'):>7}")

    print(f"\n(2) DOES THE LABEL CHANGE IF YOU ONLY LOOK AT THE FIRST 25 SEEDS?")
    ch = collections.Counter((r["cls50"], r["cls25"]) for r in rows)
    for (a, b), c in sorted(ch.items(), key=lambda x: -x[1]):
        print(f"   A50={a:<7} -> A25={b:<7} {c:>4}" + ("   (unchanged)" if a == b else "   <-- CHANGED"))

    print(f"\n(3) THE 6 HARMS: was A50's shipped pose only reachable via a LATE seed?")
    harms = [r for r in rows if r["cls50"] == "solved" and r["sB"] < ACC]
    print(f"   {'interface':<12} {'A50 sel':>8} {'A25 sel':>8} {'B sel':>7} {'shipped from':>13} "
          f"{'acc in 1st25':>13}")
    for r in sorted(harms, key=lambda r: r["sB"]):
        print(f"   {r['iface']:<12} {r['s50']:>8.3f} {r['s25']:>8.3f} {r['sB']:>7.3f} "
              f"{'LATE seed' if r['shipped_late'] else 'early seed':>13} {str(r['acc_early']):>13}")
    nl = sum(1 for r in harms if r["shipped_late"])
    print(f"   -> {nl}/{len(harms)} shipped from a seed Arm B never had; "
          f"{sum(1 for r in harms if not r['acc_early'])}/{len(harms)} had NO acceptable pose in the first 25")

    print(f"\n(4) THE 88 DECOYS: was the acceptable-but-unselected pose only in the LATE seeds?")
    dec = [r for r in rows if r["cls50"] == "decoy"]
    only_late = [r for r in dec if r["acc_late"] and not r["acc_early"]]
    print(f"   decoys: {len(dec)}")
    print(f"   acceptable pose present in first 25 seeds : {sum(1 for r in dec if r['acc_early']):>3}")
    print(f"   acceptable pose ONLY in seeds 300025-49   : {len(only_late):>3}  "
          f"<- templating could not have rescued these")
    resc = [r for r in dec if r["sB"] >= ACC]
    print(f"   rescued by Arm B: {len(resc)}  "
          f"(of which had an early acceptable pose: {sum(1 for r in resc if r['acc_early'])})")
    print(f"   RESCUE RATE on the reachable subset: {len(resc)}/{sum(1 for r in dec if r['acc_early'])}")

    print(f"\n(5) LEDGER BY COMPLEX (pseudo-replication check)")
    for lab, sel_ in (("rescued (decoy->solved)", [r for r in dec if r["sB"] >= ACC]),
                      ("broken  (solved->lost)", harms)):
        pc = collections.Counter(r["pdb"] for r in sel_)
        print(f"   {lab}: {len(sel_)} interfaces across {len(pc)} complexes  "
              f"{dict(pc)}")


# ---------------------------------------------------------------------------------------------
def batchb():
    """Batch B: templating from the OTHER 25 donors (300025-300049). Replication + saturation.

    ARMS, all 250 poses except where noted:
      A50   50 free seeds                                    -- the published baseline
      B1    free 300000-24 + templated from them              -- batch A (the original Arm B)
      B2    free 300025-49 + templated from them              -- batch B, an INDEPENDENT repeat
      POOL  all 50 free + all 50 templated (500 poses)        -- not compute-matched; saturation only

    B1 vs B2 is the replication question: is "10 rescues" reproducible or a lottery? Rescuing the SAME
    interfaces would mean templating finds a specific reachable subset; rescuing DIFFERENT ones would
    mean each draw finds its own, and pooling should keep accruing.

    ONLY COMPLETE INTERFACES are compared -- batch B runs easy-first (small, single-copy first), so a
    partial read is biased toward small complexes. Interfaces without all 25 donors are listed as
    pending rather than averaged in.
    """
    M2 = f"{EXP}/artifacts/hybrid25b"
    free = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        k = f"{r['pdb']}_{''.join(sorted((r['chain1'], r['chain2'])))}"
        free[k].append((int(r["seed"]), float(r["dockq"]), float(r["pair_iptm"])))

    def load(src):
        t, done = collections.defaultdict(list), collections.defaultdict(set)
        if not os.path.exists(src):
            return t, done
        for r in csv.DictReader(open(src)):
            try:
                t[r["interface"]].append((int(r["donor_seed"]), float(r["dockq"]), float(r["iptm"])))
                done[r["interface"]].add(int(r["donor_seed"]))
            except ValueError:
                continue
        return t, done
    t1, d1 = load(f"{M}/results_all.csv")
    t2, d2 = load(f"{M2}/results_all.csv")

    sel = lambda v: max(v, key=lambda x: x[2])[1]
    rows, pending = [], 0
    for iface in sorted(set(t2)):
        if iface not in free or len(d2[iface]) < 25:
            pending += 1
            continue
        A50 = free[iface]
        e = [x for x in A50 if FREE_LO <= x[0] <= FREE_HI]
        l = [x for x in A50 if x[0] > FREE_HI]
        B1 = e + list(t1.get(iface, []))
        B2 = l + list(t2[iface])
        POOL = A50 + list(t1.get(iface, [])) + list(t2[iface])
        cls50 = ("decoy" if sel(A50) < ACC <= max(x[1] for x in A50)
                 else ("nocov" if max(x[1] for x in A50) < ACC else "solved"))
        rows.append(dict(iface=iface, cls=cls50, a50=sel(A50),
                         b1=sel(B1) if t1.get(iface) and len(d1[iface]) >= 25 else float("nan"),
                         b2=sel(B2), pool=sel(POOL),
                         has_b1=bool(t1.get(iface)) and len(d1[iface]) >= 25))
    print(f"\nbatch B complete interfaces: {len(rows)}   pending: {pending}")
    if not rows:
        return

    print(f"\n{'class':<8} {'n':>4} {'A50':>7} {'B1':>7} {'B2':>7} {'POOL':>7}   "
          f"{'solvA50':>8} {'solvB1':>7} {'solvB2':>7} {'solvPOOL':>9}")
    for cl in ("decoy", "nocov", "solved", "ALL"):
        v = rows if cl == "ALL" else [r for r in rows if r["cls"] == cl]
        if not v:
            continue
        m = lambda k: np.nanmean([r[k] for r in v])
        n = lambda k: sum(1 for r in v if r[k] >= ACC)
        print(f"{cl:<8} {len(v):>4} {m('a50'):>7.3f} {m('b1'):>7.3f} {m('b2'):>7.3f} {m('pool'):>7.3f}   "
              f"{n('a50'):>8} {n('b1'):>7} {n('b2'):>7} {n('pool'):>9}")

    dec = [r for r in rows if r["cls"] == "decoy"]
    r1 = {r["iface"] for r in dec if r["has_b1"] and r["b1"] >= ACC}
    r2 = {r["iface"] for r in dec if r["b2"] >= ACC}
    rp = {r["iface"] for r in dec if r["pool"] >= ACC}
    print(f"\nDECOY RESCUES among the {len(dec)} decoy interfaces complete in BOTH batches")
    print(f"   batch A (B1) rescued : {len(r1):>3}  {sorted(r1)}")
    print(f"   batch B (B2) rescued : {len(r2):>3}  {sorted(r2)}")
    print(f"   both                 : {len(r1 & r2):>3}  {sorted(r1 & r2)}")
    print(f"   NEW in batch B only  : {len(r2 - r1):>3}  {sorted(r2 - r1)}")
    print(f"   lost vs batch A      : {len(r1 - r2):>3}  {sorted(r1 - r2)}")
    print(f"   union / pooled arm   : {len(r1 | r2):>3} union, {len(rp):>3} when actually pooled")


# ---------------------------------------------------------------------------------------------
def commitment(png=f"{EXP}/artifacts/commitment_spread.png"):
    """DID SELF-TEMPLATING ACTUALLY COMMIT THE TRUNK? The original motivation, never tested.

    The premise of self-templating was that showing the model its own pose forces the pair rep to
    commit to it, so the five diffusion samples of a templated run should agree with each other MORE
    than the five samples of the free run it came from. That is a direct, label-free prediction about
    WITHIN-RUN DockQ SPREAD, and it has never been checked -- every result so far has been about which
    pose gets selected, not about whether the mechanism fired.

    THE PAIRING. Templated run for donor seed d was built from ALL FIVE poses of free seed d, so the
    matched control is free seed d itself: same seed, same MSA draw distribution, the only difference
    being that its own poses came back as templates. (Run-seed matching -- comparing against free seeds
    300025-49, whose numbers the templated runs execute under -- is reported alongside, since that is
    the other defensible reading and it controls for seed identity rather than donor identity.)

    SPREAD = max-min of the 5 samples' DockQ, and SD, per run; then averaged over the 25 runs of an
    interface. Both are reported because with n=5 the range is outlier-sensitive and the SD is not.

    A LOWER templated spread confirms the mechanism. Note it would NOT by itself make the method work
    -- committing harder to a decoy is exactly the failure we measured on 9nph -- but it separates
    "the mechanism fired and the pose was wrong" from "the mechanism never fired at all".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    free = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not r["dockq"]:
            continue
        k = f"{r['pdb']}_{''.join(sorted((r['chain1'], r['chain2'])))}"
        free[k][int(r["seed"])].append(float(r["dockq"]))
    tm = collections.defaultdict(lambda: collections.defaultdict(list))
    cls = {}
    for r in csv.DictReader(open(f"{M}/results_all.csv")):
        try:
            tm[r["interface"]][int(r["donor_seed"])].append(float(r["dockq"]))
        except ValueError:
            continue
        cls[r["interface"]] = r["cls"]

    def agg(d, seeds):
        rng, sd = [], []
        for s in seeds:
            v = d.get(s, [])
            if len(v) != 5:
                continue
            rng.append(max(v) - min(v))
            sd.append(float(np.std(v, ddof=0)))
        return (float(np.mean(rng)), float(np.mean(sd)), len(rng)) if rng else (np.nan, np.nan, 0)

    rows = []
    for iface, td in tm.items():
        if iface not in free:
            continue
        fr, fs, nf = agg(free[iface], range(FREE_LO, FREE_HI + 1))          # donor seeds, free
        lr, ls, nl = agg(free[iface], range(FREE_HI + 1, FREE_HI + 26))     # run-seed matched control
        tr, ts, nt = agg(td, range(FREE_LO, FREE_HI + 1))                   # templated
        if nf < 20 or nt < 20:
            continue
        rows.append(dict(iface=iface, cls=cls.get(iface, "?"), fr=fr, fs=fs, tr=tr, ts=ts,
                         lr=lr, ls=ls, nf=nf, nt=nt, nl=nl))
    print(f"\ninterfaces with >=20 runs on both sides: {len(rows)}")

    def line(lab, a, b, na, nb):
        d = np.array(b) - np.array(a)
        try:
            from scipy.stats import wilcoxon
            p = wilcoxon(d).pvalue
        except Exception:
            p = float("nan")
        print(f"   {lab:<34} free {np.mean(a):.4f}   templated {np.mean(b):.4f}   "
              f"diff {np.mean(d):+.4f}   lower in {int((d<0).sum())}/{len(d)}   p={p:.2g}")

    print(f"\nMEAN WITHIN-RUN DockQ SPREAD (averaged over the 25 runs of each interface)")
    line("range, vs DONOR seeds 300000-24", [r["fr"] for r in rows], [r["tr"] for r in rows], 0, 0)
    line("SD,    vs DONOR seeds 300000-24", [r["fs"] for r in rows], [r["ts"] for r in rows], 0, 0)
    ok = [r for r in rows if r["nl"] >= 20]
    if ok:
        line("range, vs RUN seeds 300025-49", [r["lr"] for r in ok], [r["tr"] for r in ok], 0, 0)

    print(f"\nBY CLASS (range, vs donor seeds)")
    print(f"   {'class':<8} {'n':>4} {'free':>8} {'templated':>10} {'diff':>8} {'lower':>9}")
    for cl in ("decoy", "nocov", "solved", "?"):
        v = [r for r in rows if r["cls"] == cl]
        if not v:
            continue
        a = np.array([r["fr"] for r in v]); b = np.array([r["tr"] for r in v])
        print(f"   {cl:<8} {len(v):>4} {a.mean():>8.4f} {b.mean():>10.4f} {(b-a).mean():>+8.4f} "
              f"{str(int((b<a).sum()))+'/'+str(len(v)):>9}")

    COL = {"decoy": "#b03a48", "solved": "#2f7fb8", "nocov": "#8a8a8a", "?": "#d9a441"}
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for j, (key, lab) in enumerate(((("fr", "tr"), "within-run DockQ range (max-min)"),
                                    (("fs", "ts"), "within-run DockQ SD"))):
        (ka, kb) = key
        for cl in ("decoy", "solved", "nocov", "?"):
            v = [r for r in rows if r["cls"] == cl]
            if not v:
                continue
            ax[j].scatter([r[ka] for r in v], [r[kb] for r in v], s=26, alpha=.75,
                          c=COL[cl], edgecolor="white", linewidth=.3, label=f"{cl} (n={len(v)})")
        m = max(max(r[ka] for r in rows), max(r[kb] for r in rows)) * 1.05
        ax[j].plot([0, m], [0, m], color="0.4", ls="--", lw=1)
        ax[j].set_xlim(0, m); ax[j].set_ylim(0, m)
        ax[j].set_xlabel(f"FREE donor seeds 300000-24 — {lab}")
        ax[j].set_ylabel(f"TEMPLATED runs — {lab}")
        ax[j].set_title("below the line = templating COMMITTED harder")
        ax[j].legend(frameon=False, fontsize=8)
    fig.suptitle("Did self-templating reduce diffusion-sample spread?  "
                 f"one dot = one interface (n={len(rows)}), mean over its 25 runs")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"\nwrote {png}")


# ---------------------------------------------------------------------------------------------
def confvsbest(png=f"{EXP}/artifacts/conf_vs_best_seed.png"):
    """CONFIDENT SEED vs BEST SEED, free 50-seed arm only, confident-decoy interfaces.

    Motivates "commit the pair rep to a chosen seed": if the best seed is a DIFFERENT seed from the
    most confident one, then the good answer already exists as a whole trunk state and the problem is
    purely which trunk you commit to. If they were often the same seed, there would be nothing to pick
    between and committing would be pointless.

    DEFINITIONS. A seed is summarised by its best pose, because that is what selection ships:
      confidence(seed) = max pair_iptm over its 5 diffusion samples
      quality(seed)    = max DockQ over its 5 diffusion samples
      confident seed   = argmax confidence      (what ipTM picks; its DockQ is what you get)
      best seed        = argmax quality         (the oracle seed)
    Interface class is recomputed from the free arm rather than read from the protocol file, so the
    13 interfaces missing there are included: decoy = shipped < 0.23 <= oracle.

    NO TEMPLATED POSES ANYWHERE HERE -- this is the stock 50-seed baseline, so the gap it shows is a
    property of the model, not of anything we did.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        k = f"{r['pdb']}_{''.join(sorted((r['chain1'], r['chain2'])))}"
        by[k][int(r["seed"])].append((float(r["dockq"]), float(r["pair_iptm"])))

    rows = []
    for iface, sd in by.items():
        if len(sd) < 50:
            continue
        conf = {s: max(p for _, p in v) for s, v in sd.items()}
        qual = {s: max(d for d, _ in v) for s, v in sd.items()}
        cs = max(conf, key=conf.get)
        bs = max(qual, key=qual.get)
        shipped = max(sd[cs], key=lambda x: x[1])[0]
        if not (shipped < ACC <= qual[bs]):
            continue                                  # confident-decoy interfaces only
        sd_ = lambda v: float(np.std([d for d, _ in v], ddof=0))
        rg_ = lambda v: max(d for d, _ in v) - min(d for d, _ in v)
        rows.append(dict(iface=iface, cs=cs, bs=bs, same=cs == bs,
                         cs_dq=shipped, bs_dq=qual[bs],
                         cs_ip=conf[cs], bs_ip=conf[bs],
                         cs_sd=sd_(sd[cs]), bs_sd=sd_(sd[bs]),
                         cs_rg=rg_(sd[cs]), bs_rg=rg_(sd[bs]),
                         rank_of_best=sorted(conf, key=conf.get, reverse=True).index(bs) + 1))
    print(f"\nconfident-decoy interfaces with a full 50 seeds: {len(rows)}")
    same = sum(1 for r in rows if r["same"])
    print(f"   confident seed IS the best seed: {same}/{len(rows)}")
    print(f"   shipped DockQ  mean {np.mean([r['cs_dq'] for r in rows]):.3f}   "
          f"best DockQ mean {np.mean([r['bs_dq'] for r in rows]):.3f}")
    print(f"   ipTM of confident seed mean {np.mean([r['cs_ip'] for r in rows]):.3f}   "
          f"ipTM of best seed mean {np.mean([r['bs_ip'] for r in rows]):.3f}")
    rk = np.array([r["rank_of_best"] for r in rows])
    print(f"   CONFIDENCE RANK OF THE BEST SEED (1 = ipTM would have picked it):")
    print(f"      median {int(np.median(rk))} of 50   mean {rk.mean():.1f}   "
          f"in top-5: {int((rk<=5).sum())}/{len(rk)}   in top-10: {int((rk<=10).sum())}/{len(rk)}   "
          f"bottom half: {int((rk>25).sum())}/{len(rk)}")

    print(f"\n   WITHIN-SEED DockQ SPREAD across the 5 poses")
    for lab, ka, kb in (("SD   ", "cs_sd", "bs_sd"), ("range", "cs_rg", "bs_rg")):
        a = np.array([r[ka] for r in rows]); b = np.array([r[kb] for r in rows])
        try:
            from scipy.stats import wilcoxon
            pv = wilcoxon(b - a).pvalue
        except Exception:
            pv = float("nan")
        print(f"      {lab}  confident seed {a.mean():.4f}   best seed {b.mean():.4f}   "
              f"diff {(b-a).mean():+.4f}   best-seed wider in {int((b>a).sum())}/{len(a)}   p={pv:.2g}")
    print(f"      confident seed FROZEN (SD < 0.005): {int((np.array([r['cs_sd'] for r in rows])<0.005).sum())}/{len(rows)}"
          f"   best seed frozen: {int((np.array([r['bs_sd'] for r in rows])<0.005).sum())}/{len(rows)}")

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6))
    for j2, (ka, kb, lab) in enumerate(((("cs_sd"), ("bs_sd"), "DockQ SD across the 5 poses"),
                                        (("cs_rg"), ("bs_rg"), "DockQ range (max-min) across the 5 poses"))):
        a = ax[j2]
        a.scatter([r[ka] for r in rows], [r[kb] for r in rows], s=34, alpha=.8,
                  c="#b03a48", edgecolor="white", linewidth=.4)
        m = max(max(r[ka] for r in rows), max(r[kb] for r in rows)) * 1.05
        a.plot([0, m], [0, m], color="0.4", ls="--", lw=1)
        a.set_xlim(0, m); a.set_ylim(0, m)
        a.set_xlabel(f"MOST CONFIDENT seed — {lab}")
        a.set_ylabel(f"BEST seed — {lab}")
        a.set_title("above the line = the good seed is the less committed one")
    fig.suptitle(f"Within-seed diffusion spread: confident vs best seed\n"
                 f"{len(rows)} confident-decoy interfaces, 50 free seeds")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


# ---------------------------------------------------------------------------------------------
def spread_vs_quality(png=f"{EXP}/artifacts/seed_spread_vs_quality.png"):
    """Per-seed diffusion spread against that seed's mean pose quality, split by interface class.

    ONE DOT = ONE SEED of one interface (50 free seeds per interface, 5 poses each). x is the seed's
    MEAN DockQ over its 5 poses, y is its spread (max-min). No templated poses -- stock 50-seed arm.

    THE QUESTION. On confident-decoy interfaces the seed that holds the answer has ~9x the spread of
    the seed ipTM ships (0.138 vs 0.015), while on solved interfaces the same ratio is only 1.7x and
    the narrow confident seed is already correct. So spread looks informative only where the default
    attractor is wrong. Plotting every seed rather than just the two extremes shows whether that is a
    property of the whole population or an artifact of comparing two hand-picked seeds.

    Shape to look for: on solved interfaces, high-quality seeds should be able to be NARROW (the good
    basin is the attractor, so a committed seed lands in it). On decoys, high-quality seeds should be
    forced WIDE, because reaching a good pose requires not having settled into the default well.
    Colour is the seed's max pair_iptm, so the third variable -- what confidence rewards -- is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(f"{P}/dockq_pxm2025.csv")):
        if not r["dockq"] or not r["pair_iptm"]:
            continue
        k = f"{r['pdb']}_{''.join(sorted((r['chain1'], r['chain2'])))}"
        by[k][int(r["seed"])].append((float(r["dockq"]), float(r["pair_iptm"])))

    pts = collections.defaultdict(list)
    for iface, sd in by.items():
        if len(sd) < 50:
            continue
        conf = {s: max(p for _, p in v) for s, v in sd.items()}
        qual = {s: max(d for d, _ in v) for s, v in sd.items()}
        cs = max(conf, key=conf.get)
        shipped = max(sd[cs], key=lambda x: x[1])[0]
        cls = ("decoy" if shipped < ACC <= max(qual.values())
               else ("nocov" if max(qual.values()) < ACC else "solved"))
        for s, v in sd.items():
            if len(v) != 5:
                continue
            d = [x[0] for x in v]
            pts[cls].append((float(np.mean(d)), max(d) - min(d), max(x[1] for x in v), s == cs))

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=True, sharey=True)
    for j, cl in enumerate(("solved", "decoy")):
        v = pts[cl]
        q = np.array([x[0] for x in v]); sp = np.array([x[1] for x in v])
        ip = np.array([x[2] for x in v]); isc = np.array([x[3] for x in v])
        a = ax[j]
        sc = a.scatter(q, sp, c=ip, cmap="viridis", s=13, alpha=.55, linewidth=0)
        a.scatter(q[isc], sp[isc], s=52, facecolor="none", edgecolor="#d1495b", linewidth=1.2,
                  label=f"the seed ipTM ships (n={int(isc.sum())})")
        a.axvline(ACC, color="0.45", ls=":", lw=1)
        r = np.corrcoef(q, sp)[0, 1]
        a.set_title(f"{'ACCEPTABLE (solved) interfaces' if cl=='solved' else 'CONFIDENT-DECOY interfaces'}"
                    f"\n{len({0})and len(v)} seeds, r(quality, spread) = {r:+.2f}")
        a.set_xlabel("seed mean DockQ over its 5 poses")
        a.legend(frameon=False, fontsize=8, loc="upper right")
    ax[0].set_ylabel("seed DockQ spread (max - min)")
    fig.colorbar(sc, ax=ax, fraction=0.03, label="seed max pair_iptm")
    fig.suptitle("Does reaching a good pose require an uncommitted seed?  one dot = one seed "
                 "(50 free seeds per interface)")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")
    for cl in ("solved", "decoy"):
        v = pts[cl]
        q = np.array([x[0] for x in v]); sp = np.array([x[1] for x in v])
        good = q >= ACC
        print(f"\n{cl}: {len(v)} seeds   r(quality,spread) = {np.corrcoef(q,sp)[0,1]:+.3f}")
        print(f"   seeds with mean DockQ >= {ACC}: {int(good.sum())}  "
              f"their mean spread {sp[good].mean():.3f}   (others {sp[~good].mean():.3f})")
        if good.any():
            print(f"   of those good seeds, NARROW (spread<0.05): {int((sp[good]<0.05).sum())}"
                  f"/{int(good.sum())}")
