#!/usr/bin/env python
"""Summarise the native-template arm, split by FAILURE MODE rather than by a single pass/fail count.

A bare "N/M rescued" hides the thing that matters: templating both chains on the native fold fixes
FOLDS, so what it can and cannot repair tells us which bottleneck each interface actually has.

For every interface we compare the free run with the native-templated run and assign:

  RESCUED             selection now picks an acceptable pose.
  SELECTION-LIMITED   coverage is healthy after templating (>=10% of poses acceptable) but the top
                      ipTM pose is still wrong. Folds were fixed; confidence was not. (8fas, 8oxw)
  COVERAGE-LIMITED    even with native folds, almost nothing acceptable is generated (<10%). The
                      docking search, not fold quality, is the wall. (7z2m, 8a44)
  THIN                fewer than 5 acceptable poses in the FREE run. These sit within sampling noise
                      of the decoy / generation-failure boundary and should not drive conclusions.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/native_arm_summary.py
Writes: artifacts/native_arm_summary.json  (and prints the tables)
"""
import json, os, sys, collections

EXP = "/home/ebrahim/confident_decoy_experiments"
CLS = json.load(open(f"{EXP}/artifacts/abag_all_101_150.json"))
RES = json.load(open(f"{EXP}/artifacts/native_arm_scores.json"))
COVERAGE_OK = 0.10          # >=10% of 250 poses acceptable = generation is not the bottleneck
THIN = 5                    # free-run acceptable poses below this => noise-limited, flag it


def classify(v):
    if v["rescued"]:
        return "RESCUED"
    return "SELECTION-LIMITED" if v["n_acceptable"] / max(v["n"], 1) >= COVERAGE_OK \
        else "COVERAGE-LIMITED"


rows = []
for k, v in sorted(RES.items()):
    rows.append(dict(interface=k, group=v["free_status"], mode=classify(v),
                     thin=(v.get("free_n_acceptable") or 0) < THIN,
                     free_n_ok=v.get("free_n_acceptable"), free_best=v.get("free_best_dockq"),
                     free_sel=v.get("free_sel_dockq"),
                     nat_n_ok=v["n_acceptable"], nat_sel=v["sel_dockq"], nat_best=v["best_dockq"],
                     contacts=CLS.get(k, {}).get("contact_residues")))

want = {k for k, v in CLS.items() if v["status"] in ("confident_decoy", "no_acceptable_pose")}
missing = sorted(want - set(RES))

for grp, title in (("confident_decoy", "RUN A - confident decoys (selection failed in the free run)"),
                   ("no_acceptable_pose", "RUN B - generation failures (no acceptable pose at all)")):
    g = [r for r in rows if r["group"] == grp]
    if not g:
        continue
    print(f"\n=== {title} ===")
    print(f"{'interface':12} {'ctc':>4} {'free n_ok':>10} {'free best':>10} "
          f"{'nat n_ok':>9} {'nat sel':>8} {'nat best':>9}  outcome")
    for r in sorted(g, key=lambda x: (x["mode"] != "RESCUED", -(x["nat_sel"] or 0))):
        flags = []
        if r["thin"]:
            flags.append("THIN")
        if r["contacts"] is not None and r["contacts"] < 10:
            flags.append(f"LABEL?{r['contacts']}res")
        print(f"{r['interface']:12} {str(r['contacts']):>4} {str(r['free_n_ok']):>10} "
              f"{r['free_best']:10.3f} {r['nat_n_ok']:9d} {r['nat_sel']:8.3f} {r['nat_best']:9.3f}  "
              f"{r['mode']}{'  [' + ','.join(flags) + ']' if flags else ''}")
    c = collections.Counter(r["mode"] for r in g)
    solid = [r for r in g if not r["thin"]]
    print(f"  totals: " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"  rescued {sum(1 for r in g if r['mode']=='RESCUED')}/{len(g)}   "
          f"(excluding THIN cases: {sum(1 for r in solid if r['mode']=='RESCUED')}/{len(solid)})")

if missing:
    print(f"\nNOT SCORED ({len(missing)}): {missing}")
json.dump({"coverage_threshold": COVERAGE_OK, "thin_threshold": THIN,
           "rows": rows, "not_scored": missing},
          open(f"{EXP}/artifacts/native_arm_summary.json", "w"), indent=1)
print(f"\nWROTE {EXP}/artifacts/native_arm_summary.json")
