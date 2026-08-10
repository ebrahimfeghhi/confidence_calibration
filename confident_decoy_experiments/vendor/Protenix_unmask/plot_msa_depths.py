"""Plot histogram of paired and unpaired MSA depths across foldbench structures."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

CSV = Path("foldbench_abag/msa_depths.csv")
OUT = Path("results/msa_depths_hist.png")

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
