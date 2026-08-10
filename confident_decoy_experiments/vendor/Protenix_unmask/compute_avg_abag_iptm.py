"""
Post-processing: add avg_abag_iptm and new_ranking_score columns to the DockQ CSV.

Reads chain_mapping from the CSV, uses it to index chain_pair_iptm from the
confidence JSON, and computes:

    avg_abag_iptm  = mean of chain_pair_iptm[ab_idx][ag_idx] over all Ab-Ag interfaces
    new_ranking_score = 0.8 * avg_abag_iptm + 0.2 * ptm - 100 * has_clash

Usage:
    python compute_avg_abag_iptm.py
    python compute_avg_abag_iptm.py --in_csv foldbench_abag/dockq_ost_predictions_template_all_samples.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PRED_DIR = Path("foldbench_abag/predictions_template")
IN_CSV   = Path("foldbench_abag/dockq_ost_predictions_template_all_samples.csv")


def find_model_chain(chain_mapping, ref_chain):
    if ref_chain in chain_mapping:
        return chain_mapping[ref_chain]
    for k, v in chain_mapping.items():
        if k.split("-")[0] == ref_chain:
            return v
    return None


def model_chain_to_idx(chain_str):
    letter = chain_str.split("-")[0].upper()
    return ord(letter) - ord("A")


def read_confidence(pred_dir, pdb_base, seed, sample):
    path = pred_dir / pdb_base / f"seed_{seed}" / "predictions" / \
           f"{pdb_base}_summary_confidence_sample_{sample}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("chain_pair_iptm"), d.get("ptm"), d.get("has_clash", False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv",   default=str(IN_CSV))
    parser.add_argument("--out_csv",  default=None)
    parser.add_argument("--pred_dir", default=str(PRED_DIR))
    args = parser.parse_args()

    in_csv   = Path(args.in_csv)
    out_csv  = Path(args.out_csv) if args.out_csv else \
               in_csv.parent / (in_csv.stem + "_scored.csv")
    pred_dir = Path(args.pred_dir)

    print(f"Reading {in_csv}")
    rows = []
    fieldnames = None
    with open(in_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    print(f"  {len(rows)} rows loaded")

    if "chain_mapping" not in fieldnames:
        print("ERROR: chain_mapping column missing — rerun eval first")
        return

    # Group rows by (pdb_base, seed, sample)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        pdb_base = row["assembly_id"].split("-")[0]
        try:
            seed   = int(row["seed"])
            sample = int(row["sample"])
        except (ValueError, KeyError):
            continue
        groups[(pdb_base, seed, sample)].append(i)

    nan = float("nan")
    avg_map = {}       # row_idx -> avg_abag_iptm
    nrs_map = {}       # row_idx -> new_ranking_score
    pair_iptm_map = {}  # row_idx -> per-interface iptm

    missing_conf = 0
    missing_mapping = 0

    for (pdb_base, seed, sample), indices in groups.items():
        conf = read_confidence(pred_dir, pdb_base, seed, sample)
        if conf is None or conf[0] is None:
            missing_conf += 1
            for i in indices:
                avg_map[i] = nan
                nrs_map[i] = nan
            continue

        chain_pair_iptm, ptm, has_clash = conf

        # Get chain_mapping (first non-empty one in the group)
        chain_mapping = {}
        for i in indices:
            cm_str = rows[i].get("chain_mapping", "")
            if cm_str:
                try:
                    chain_mapping = json.loads(cm_str)
                    break
                except Exception:
                    pass

        if not chain_mapping:
            missing_mapping += 1
            for i in indices:
                avg_map[i] = nan
                nrs_map[i] = nan
            continue

        # Compute per-interface iptm and aggregate avg for this group
        pair_map_local = {}  # row_idx -> per-interface iptm
        iptm_vals = []
        for i in indices:
            chain1 = rows[i]["chain1"]
            chain2 = rows[i]["chain2"]
            mc1 = find_model_chain(chain_mapping, chain1)
            mc2 = find_model_chain(chain_mapping, chain2)
            if mc1 is None or mc2 is None:
                pair_map_local[i] = nan
                continue
            idx1 = model_chain_to_idx(mc1)
            idx2 = model_chain_to_idx(mc2)
            n = len(chain_pair_iptm)
            if idx1 < n and idx2 < len(chain_pair_iptm[idx1]):
                val = chain_pair_iptm[idx1][idx2]
                pair_map_local[i] = val
                iptm_vals.append(val)
            else:
                pair_map_local[i] = nan

        avg = sum(iptm_vals) / len(iptm_vals) if iptm_vals else nan
        if avg == avg and ptm is not None:
            new_rs = 0.8 * avg + 0.2 * ptm - (100.0 if has_clash else 0.0)
        else:
            new_rs = nan

        for i in indices:
            avg_map[i] = avg
            nrs_map[i] = new_rs
            pair_iptm_map[i] = pair_map_local.get(i, nan)

    print(f"  Missing confidence JSONs: {missing_conf} groups")
    print(f"  Missing chain_mapping:    {missing_mapping} groups")

    new_fieldnames = list(fieldnames) + ["avg_abag_iptm", "new_ranking_score", "pair_iptm"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            row["avg_abag_iptm"]    = avg_map.get(i, nan)
            row["new_ranking_score"] = nrs_map.get(i, nan)
            row["pair_iptm"]       = pair_iptm_map.get(i, nan)
            writer.writerow(row)

    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
