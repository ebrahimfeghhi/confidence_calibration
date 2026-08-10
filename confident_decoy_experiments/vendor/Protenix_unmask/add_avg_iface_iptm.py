"""
Read the existing dockq CSV (one row per interface/seed, selected by ranking_score),
add an ranking_score_new column computed from the confidence JSONs, and write a new CSV.

ranking_score_new for a seed = max over samples of (mean chain_pair_iptm across all Ab-Ag pairs),
consistent with how ranking_score is stored (score of the best sample).
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

IN_CSV   = Path("foldbench_abag/dockq_ost_predictions_template.csv")
OUT_CSV  = Path("foldbench_abag/dockq_ost_predictions_template_ranking_score_new.csv")
PRED_DIR = Path("foldbench_abag/predictions_template")
TARGETS  = Path("/home/ebrahim/FoldBench/targets/interface_antibody_antigen.csv")


def chain_to_asym_id(chain_id: str) -> int:
    base = chain_id.split("-")[0]
    idx = 0
    for ch in base:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


# Build pdb_base -> list of (chain1, chain2) from targets CSV
pdb_interfaces = defaultdict(list)
with open(TARGETS) as f:
    for row in csv.DictReader(f):
        pdb_base = row["pdb_id"].split("-")[0]
        pdb_interfaces[pdb_base].append((row["interface_chain_id_1"], row["interface_chain_id_2"]))


def best_ranking_score_new(pdb_base: str, seed: int) -> float:
    """Max ranking_score_new across all 5 samples for this (pdb_base, seed)."""
    seed_dir = PRED_DIR / pdb_base / f"seed_{seed}" / "predictions"
    pairs = pdb_interfaces[pdb_base]
    pair_indices = [(chain_to_asym_id(c1), chain_to_asym_id(c2)) for c1, c2 in pairs]

    best = float("-inf")
    for conf_file in sorted(seed_dir.glob(f"{pdb_base}_summary_confidence_sample_*.json")):
        try:
            conf = json.loads(conf_file.read_text())
            matrix = conf.get("chain_pair_iptm", [])
            scores = [matrix[i][j] for i, j in pair_indices
                      if i < len(matrix) and j < len(matrix[i])]
            avg_iface = sum(scores) / len(scores) if scores else float("-inf")
            full_score = (0.8 * avg_iface
                          + 0.2 * conf.get("ptm", 0)
                          - 100 * conf.get("has_clash", 0))
            if full_score > best:
                best = full_score
        except Exception:
            pass
    return best


# Read existing CSV, add ranking_score_new, write new CSV
rows = []
with open(IN_CSV) as f:
    for row in csv.DictReader(f):
        pdb_base = row["assembly_id"].split("-")[0]
        seed = int(row["seed"])
        row["ranking_score_new"] = best_ranking_score_new(pdb_base, seed)
        rows.append(row)

print(f"Processed {len(rows)} rows")

fieldnames = ["assembly_id", "chain1", "chain2", "seed", "ranking_score_new", "dockq"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"Written to {OUT_CSV}")
