"""
Slopegraph: per-interface pass rate at N=1 vs N=50.
N=1  = fraction of individual seeds with DockQ >= threshold
N=50 = 1 if best-by-ranking-score seed exceeds threshold, else 0
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import RESULTS_DIR

CSV       = Path("foldbench_abag/dockq_ost_predictions_template_all_samples.csv")
OUT       = RESULTS_DIR / "tts_slopegraph.png"
THRESHOLD = 0.23

# Group by interface+seed, pick best sample by pair_iptm, then collect per-seed DockQ
by_iface_seeds = defaultdict(lambda: defaultdict(list))
with open(CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            dq = float(row["dockq"])
            rs = float(row["pair_iptm"])
            seed = int(row["seed"])
        except (ValueError, KeyError):
            continue
        if dq != dq or rs != rs:
            continue
        key = (row["assembly_id"], row["chain1"], row["chain2"])
        by_iface_seeds[key][seed].append((rs, dq))

by_iface = defaultdict(list)
for key, seeds in by_iface_seeds.items():
    for seed, samples in seeds.items():
        best_rs, best_dq = max(samples, key=lambda x: x[0])
        by_iface[key].append((best_rs, best_dq))

points = []
for key, rows in by_iface.items():
    p1  = sum(dq for _, dq in rows) / len(rows)
    p50 = max(rows, key=lambda x: x[0])[1]
    points.append((p1, p50))

p1s  = [p[0] for p in points]
p50s = [p[1] for p in points]

# Color by gain
gains = [p50 - p1 for p1, p50 in points]
norm  = plt.Normalize(min(gains), max(gains))
cmap  = cm.RdYlGn

fig, ax = plt.subplots(figsize=(6, 9))

for p1, p50, gain in zip(p1s, p50s, gains):
    color = cmap(norm(gain))
    ax.plot([0, 1], [p1, p50], color=color, alpha=0.5, linewidth=0.8)

# Scatter dots on top
ax.scatter([0] * len(p1s),  p1s,  color="steelblue", s=18, zorder=3, alpha=0.7)
ax.scatter([1] * len(p50s), p50s, color="darkorange", s=18, zorder=3, alpha=0.7)

ax.set_xticks([0, 1])
ax.set_xticklabels(["N = 1\n(mean DockQ)", "N = 50\n(best-of-50 DockQ)"], fontsize=12)
ax.set_ylabel("DockQ", fontsize=11)
ax.set_title(f"Per-interface TTS gain  ({len(points)} interfaces)", fontsize=12)
ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, axis="y", alpha=0.3)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Gain (N=50 − N=1)", fraction=0.03, pad=0.04)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
