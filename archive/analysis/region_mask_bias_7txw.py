"""Region-level mask bias for 7txw (CPU-only): do good masks favor/avoid the
epitope, paratope, or CDRs?

Loads the precomputed mask matrix (analysis/save_masks_7txw.py -> mask_matrix_7txw.npz),
labels seeds good/bad by oracle best-of-5 A|C DockQ, and for each structural region
reports the mean fraction of that region masked by good vs bad seeds vs chance (~0.10).
Minimal stats: fractions + deltas; a label-shuffle p per region is printed as a footnote.

Regions: epitope (antigen interface, <8A), paratope (antibody interface), CDRs (ANARCI/IMGT
via a working anarci interpreter), antibody framework (non-CDR), antigen non-epitope, whole
chains, all (chance). CDRs are skipped gracefully if no working anarci is found.

Usage: python analysis/region_mask_bias_7txw.py [--anarci_py /path/to/python]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))
from paths import CSV_DIR, RESULTS_DIR, GROUND_TRUTHS
import analyze_masks as am

DEFAULT_ANARCI_PY = "/home/ebrahim/miniconda3/envs/anarci/bin/python"


def oracle_dockq(csv_path, prefix, c1, c2):
    best = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["assembly_id"].startswith(prefix):
                continue
            if {row["chain1"], row["chain2"]} != {c1, c2}:
                continue
            try:
                seed, dq = int(row["seed"]), float(row["dockq"])
            except (ValueError, KeyError):
                continue
            best[seed] = max(best.get(seed, -1.0), dq)
    return best


def get_cdr_residues(model_seqs, antibody_asyms, anarci_py):
    """{(asym, res0)} CDR residues via ANARCI, or None if it's unavailable."""
    if not Path(anarci_py).exists():
        print(f"  (anarci interpreter not found: {anarci_py} — skipping CDRs)")
        return None
    seqs = [model_seqs[a] for a in antibody_asyms]
    helper = Path(__file__).resolve().parent / "anarci_cdr.py"
    env = {**os.environ, "PATH": f"{Path(anarci_py).parent}:{os.environ.get('PATH','')}"}
    r = subprocess.run([anarci_py, str(helper), *seqs],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not r.stdout.strip():
        print(f"  (ANARCI failed — skipping CDRs)\n  stderr: {r.stderr[-400:]}")
        return None
    data = json.loads(r.stdout.strip().splitlines()[-1])
    out = set()
    for i, a in enumerate(antibody_asyms):
        for idx in data.get(str(i), []):
            out.add((a, idx))
    return out


def perm_p(frac_per_seed, good_idx, bad_idx, n_perm=10000, seed=0):
    v = frac_per_seed
    obs = v[good_idx].mean() - v[bad_idx].mean()
    pool = np.concatenate([good_idx, bad_idx])
    ng = len(good_idx)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(pool)
        null[k] = v[perm[:ng]].mean() - v[perm[ng:]].mean()
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anarci_py", default=DEFAULT_ANARCI_PY)
    ap.add_argument("--good", type=float, default=0.7)
    ap.add_argument("--bad", type=float, default=0.23)
    args = ap.parse_args()

    # --- masks (from disk, CPU) ---
    d = np.load(CSV_DIR / "mask_matrix_7txw.npz")
    M = d["M"]
    residues = [tuple(int(x) for x in r) for r in d["residues"]]
    seeds = list(int(s) for s in d["seeds"])
    col_of = {r: i for i, r in enumerate(residues)}
    p_chance = float(M.mean())

    # --- outcomes / labels ---
    dq = oracle_dockq(CSV_DIR / "dockq_ost_7txw_tts2060_all_samples.csv", "7txw", "A", "C")
    good_idx = np.array([i for i, s in enumerate(seeds) if dq.get(s, -1) >= args.good])
    bad_idx = np.array([i for i, s in enumerate(seeds) if 0 <= dq.get(s, -1) < args.bad])
    print(f"good (>= {args.good}) = {len(good_idx)}   bad (< {args.bad}) = {len(bad_idx)}   "
          f"chance masked frac = {p_chance:.4f}\n")

    # --- structural region sets ---
    spi, model_seqs = am.build_spi("7txw")
    gt = GROUND_TRUTHS / "7txw-assembly1.cif"
    epitope, paratope = set(), set()
    for ab in ("A", "B"):
        s1, s2 = am.get_interface_ref(gt, model_seqs, ab, "C")  # s1 on ab, s2 on C
        paratope |= {(ord(l) - 65, p) for l, p in s1}
        epitope |= {(ord(l) - 65, p) for l, p in s2}

    A_all = {r for r in residues if r[0] == 0}
    B_all = {r for r in residues if r[0] == 1}
    C_all = {r for r in residues if r[0] == 2}
    ab_all = A_all | B_all
    cdr = get_cdr_residues(model_seqs, [0, 1], args.anarci_py)

    regions = {
        "epitope (Ag iface)": epitope,
        "paratope (Ab iface)": paratope,
    }
    if cdr is not None:
        regions["CDRs (Ab)"] = cdr
        regions["Ab framework (non-CDR)"] = ab_all - cdr
    else:
        regions["Ab non-interface"] = ab_all - paratope
    regions.update({
        "Ag non-epitope": C_all - epitope,
        "antibody all": ab_all,
        "antigen all": C_all,
        "all (chance)": set(residues),
    })

    # --- per-region fractions, good vs bad vs chance ---
    rows = []
    print(f"{'region':24s} {'size':>5s} {'good':>7s} {'bad':>7s} {'chance':>7s} "
          f"{'Δg-ch':>7s} {'Δg-b':>7s} {'perm_p':>8s}")
    for name, S in regions.items():
        cols = [col_of[r] for r in S if r in col_of]
        if not cols:
            continue
        sub = M[:, cols]
        frac_seed = sub.mean(1)
        fg = float(frac_seed[good_idx].mean())
        fb = float(frac_seed[bad_idx].mean())
        pv = (float("nan") if name == "all (chance)"
              else perm_p(frac_seed, good_idx, bad_idx))
        rows.append({"region": name, "size": len(cols), "frac_good": fg,
                     "frac_bad": fb, "frac_chance": p_chance,
                     "d_good_chance": fg - p_chance, "d_good_bad": fg - fb,
                     "perm_p": pv})
        star = " *" if (pv == pv and pv < 0.05) else ""
        print(f"{name:24s} {len(cols):5d} {fg:7.3f} {fb:7.3f} {p_chance:7.3f} "
              f"{fg - p_chance:+7.3f} {fg - fb:+7.3f} {pv:8.3f}{star}")

    out_csv = CSV_DIR / "region_mask_bias_7txw.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved {out_csv}")
    _plot(rows, p_chance, len(good_idx), len(bad_idx))


def _plot(rows, p_chance, ng, nb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["region"] != "all (chance)"]
    names = [r["region"] for r in rows]
    fg = [r["frac_good"] for r in rows]
    fb = [r["frac_bad"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, fg, 0.4, label=f"good ({ng})", color="seagreen")
    ax.bar(x + 0.2, fb, 0.4, label=f"bad ({nb})", color="firebrick")
    ax.axhline(p_chance, ls="--", color="gray", lw=1.2, label=f"chance {p_chance:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("mean fraction of region masked")
    ax.set_title("7txw A|C: does a good mask favor/avoid each region? (oracle best-of-5)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out = _p / "analysis" / "figs" / "region_mask_bias_7txw.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
