"""
Scatter: mean DockQ at N=1 (x) vs best-of-50 DockQ (y), one dot per interface.
Diagonal = no gain from TTS.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="foldbench_abag/dockq_ost_predictions_template_all_samples_scored.csv")
parser.add_argument("--out", default=str(RESULTS_DIR / "tts_scatter.png"))
parser.add_argument("--threshold", type=float, default=0.23)
args = parser.parse_args()

CSV = Path(args.csv)
OUT = Path(args.out)
THRESHOLD = args.threshold

by_iface = defaultdict(lambda: defaultdict(list))  # key -> seed -> [(score, dq)]
with open(CSV) as f:
    reader = csv.DictReader(f)
    score_col = next((c for c in ["pair_iptm", "new_ranking_score", "ranking_score_new", "avg_abag_iptm", "ranking_score"]
                      if c in reader.fieldnames), "ranking_score")
    for row in reader:
        try:
            dq = float(row["dockq"])
            rs = float(row[score_col])
            seed = int(row["seed"])
        except (ValueError, KeyError):
            continue
        if dq != dq or rs != rs:
            continue
        key = (row["assembly_id"], row["chain1"], row["chain2"])
        by_iface[key][seed].append((rs, dq))

# Pick best sample per seed by score, then build per-interface list
by_iface_best = defaultdict(list)
for key, seeds in by_iface.items():
    for seed, samples in seeds.items():
        best_rs, best_dq = max(samples, key=lambda x: x[0])
        by_iface_best[key].append((best_rs, best_dq))
by_iface = by_iface_best

n1, n50 = [], []
for rows in by_iface.values():
    n1.append(sum(dq for _, dq in rows) / len(rows))
    n50.append(max(rows, key=lambda x: x[0])[1])

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(n1, n50, s=25, alpha=0.7, color="steelblue", edgecolors="white", linewidths=0.4)

# Diagonal
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="no gain")

# Threshold lines
ax.axhline(THRESHOLD, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
ax.axvline(THRESHOLD, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

ax.set_xlabel("Mean DockQ  (N = 1)", fontsize=12)
ax.set_ylabel("Best DockQ  (N = 50)", fontsize=12)
ax.set_title(f"TTS gain per interface  ({len(n1)} interfaces)", fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
