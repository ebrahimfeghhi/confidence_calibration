"""
Scatter: mean DockQ at N=1 (x) vs best-of-50 DockQ (y), colored by total MSA depth.
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

DOCKQ_CSV = Path("foldbench_abag/dockq_ost_predictions_template_all_samples_scored.csv")
MSA_CSV   = Path("foldbench_abag/msa_depths.csv")
OUT       = Path("results/tts_scatter_msa.png")
THRESHOLD = 0.23

# Load MSA depths keyed by pdb_id
msa_depth = {}
with open(MSA_CSV) as f:
    for row in csv.DictReader(f):
        msa_depth[row["pdb_id"]] = (
            int(row["prot_paired_num_alignments"]) +
            int(row["prot_unpaired_num_alignments"])
        )

# Group by interface+seed, pick best sample by pair_iptm, then collect per-seed DockQ
by_iface_seeds = defaultdict(lambda: defaultdict(list))
with open(DOCKQ_CSV) as f:
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

n1, n50, depths = [], [], []
for (assembly_id, c1, c2), rows in by_iface.items():
    pdb_id = assembly_id.split("-")[0]
    if pdb_id not in msa_depth:
        continue
    n1.append(sum(dq for _, dq in rows) / len(rows))
    n50.append(max(rows, key=lambda x: x[0])[1])
    depths.append(msa_depth[pdb_id])

depths = np.array(depths)
norm = plt.Normalize(depths.min(), depths.max())
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(7, 6))

sc = ax.scatter(n1, n50, c=depths, cmap=cmap, norm=norm,
                s=25, alpha=0.85, edgecolors="white", linewidths=0.4)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="no gain")
ax.axhline(THRESHOLD, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
ax.axvline(THRESHOLD, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

plt.colorbar(sc, ax=ax, label="Total MSA depth (paired + unpaired)")

ax.set_xlabel("Mean DockQ  (N = 1)", fontsize=12)
ax.set_ylabel("Best DockQ  (N = 50)", fontsize=12)
ax.set_title(f"TTS gain vs MSA depth  ({len(n1)} interfaces)", fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
