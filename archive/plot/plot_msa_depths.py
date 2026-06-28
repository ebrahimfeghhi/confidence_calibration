"""Plot histogram of paired and unpaired MSA depths across foldbench structures."""
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import JSON_DIR, RESULTS_DIR

CSV = JSON_DIR.parent / "msa_depths.csv"
OUT = RESULTS_DIR / "msa_depths_hist.png"

total = []
with open(CSV) as f:
    for row in csv.DictReader(f):
        total.append(int(row["prot_paired_num_alignments"]) + int(row["prot_unpaired_num_alignments"]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(total, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
ax.axvline(sum(total)/len(total), color="black", linestyle="--", linewidth=1.2,
           label=f"mean={sum(total)/len(total):.0f}")
ax.set_xlabel("Total MSA depth (paired + unpaired)")
ax.set_ylabel("# structures")
ax.set_title("Total MSA depth across FoldBench Ab-Ag structures")
ax.legend()
ax.grid(True, alpha=0.3)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
print(f"min={min(total)}  mean={sum(total)/len(total):.0f}  max={max(total)}")
