#!/usr/bin/env python
"""Summarise one fold+commit unit: did committing to each pose separate the arms' confidences?

The donor row is the ordinary 10-recycle fold -- its five poses are what the p-arms were shown. The
none row is a stock 11th recycle with real templates, which is what recycle 11 would have done
anyway; without it, any change could be the extra recycle rather than the pose.

Reads artifacts/commit11/results.csv and, if present, checks that z_init/s_init are identical between
the control and a templated arm -- the loop-invariance the design assumes.
"""
import argparse, collections, csv, os
import numpy as np

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
ap = argparse.ArgumentParser()
ap.add_argument("--iface", required=True)
ap.add_argument("--seed", required=True)
A = ap.parse_args()

rows = [r for r in csv.DictReader(open(f"{EXP}/artifacts/commit11/results.csv"))
        if r["iface"] == A.iface and r["seed"] == str(A.seed)]
by = collections.defaultdict(list)
for r in rows:
    try:
        by[r["arm"]].append((int(r["sample"]), float(r["dockq"]),
                             float(r["iptm"]) if r["iptm"] else np.nan))
    except ValueError:
        pass

don = {s: (d, p) for s, d, p in sorted(by.get("donor", []))}
print(f"\n{A.iface} seed {A.seed}")
print(f"\nDONOR poses (ordinary 10-recycle fold) -- these are what each arm was shown")
for s in sorted(don):
    print(f"   pose {s}  DockQ {don[s][0]:.3f}  ipTM {don[s][1]:.3f}")
if don:
    d = [v[0] for v in don.values()]; p = [v[1] for v in don.values()]
    print(f"   -> DockQ span {max(d)-min(d):.3f}   ipTM span {np.nanmax(p)-np.nanmin(p):.3f}")

print(f"\nARMS (one further recycle from the same trunk state)")
print(f"{'arm':<6} {'shown pose':>10} {'its DockQ':>10} {'n':>2} {'DockQ mean':>11} {'max':>7} "
      f"{'spread':>7} {'ipTM mean':>10} {'max':>7}")
xs, ys = [], []
for tag in ["none"] + [f"p{i}" for i in range(10)]:
    v = by.get(tag)
    if not v:
        continue
    dq = np.array([x[1] for x in v]); ip = np.array([x[2] for x in v], dtype=float)
    shown = int(tag[1:]) if tag.startswith("p") else None
    sd = don.get(shown, (np.nan,))[0] if shown is not None else np.nan
    print(f"{tag:<6} {('-' if shown is None else shown):>10} "
          f"{('' if np.isnan(sd) else f'{sd:.3f}'):>10} {len(v):>2} {dq.mean():>11.3f} "
          f"{dq.max():>7.3f} {dq.max()-dq.min():>7.3f} {np.nanmean(ip):>10.3f} {np.nanmax(ip):>7.3f}")
    if shown is not None and not np.isnan(sd) and np.isfinite(np.nanmean(ip)):
        xs.append(sd); ys.append(float(np.nanmean(ip)))

if len(xs) >= 3:
    print(f"\nDOES ARM CONFIDENCE TRACK THE POSE IT WAS SHOWN?")
    print(f"   corr(shown-pose DockQ, arm ipTM) = {np.corrcoef(xs, ys)[0,1]:+.3f} over {len(xs)} arms")
    print(f"   arm ipTM span {max(ys)-min(ys):.3f}")
    print(f"   most confident arm was shown DockQ {xs[int(np.argmax(ys))]:.3f}; "
          f"best pose available was {max(xs):.3f}")

d = f"{EXP}/artifacts/commit11/{A.iface}_s{A.seed}"
a, b = f"{d}/zinit_none.npz", f"{d}/zinit_p0.npz"
if os.path.exists(a) and os.path.exists(b):
    Za, Zb = np.load(a), np.load(b)
    print(f"\nLOOP-INVARIANCE CHECK (z_init/s_init should be identical with and without a template)")
    for k in ("z_init", "s_init", "s_inputs"):
        if k in Za.files and k in Zb.files:
            dd = float(np.abs(Za[k].astype(np.float64) - Zb[k].astype(np.float64)).max())
            print(f"   {k:<9} max|control - templated| = {dd:.3e}  "
                  f"{'IDENTICAL' if dd == 0 else 'DIFFERS'}")
