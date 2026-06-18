"""
Why do certain input masks rescue hard antibody-antigen interfaces?

Generalizes analyze_masks_7yxu.py from one complex to the whole "rescue set"
(interfaces where the no-mask baseline DockQ is low but some seed skyrockets).

Hypothesis: a good mask hits non-interface "distractor" regions MORE than the
interface, forcing the network onto the true interface. Predictions:
  (1) masking interface residues hurts DockQ  -> negative frac_iface_masked vs DockQ
  (2) masking certain non-interface residues helps -> per-residue sensitivity map

ESMFold2-Fast is single-sequence; input token masking (lm_mask_pct) is the source
of test-time-scaling diversity. The mask is FIXED within a seed (shared by its 5
diffusion samples) and deterministically reproducible from the seed via _seed_context.

Outputs (under RESULTS_DIR / CSV_DIR):
  - csv_files/mask_features_per_seed.csv      (interface, seed) frac masked by region + DockQ
  - csv_files/mask_sensitivity_per_residue.csv (interface, residue) sensitivity
  - mask_rescue_region_effect.png             pooled region-level effect
  - mask_rescue_scatter.png                   frac_iface_masked vs DockQ, all rescue ifaces
  - markdown_docs/why_masks_rescue.md         findings

Usage (needs the esm venv + a GPU; CUDA RNG must match the real runs):
    CUDA_VISIBLE_DEVICES=7 ~/esm/.venv/bin/python analysis/analyze_masks.py
"""
import csv
import json
import string
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import JSON_DIR, GROUND_TRUTHS, TARGETS_CSV, CSV_DIR, RESULTS_DIR

# torch + esm are only needed for the GPU mask-reproduction path. --summary-only
# reruns the stats/report from the CSVs under plain python, so import lazily.
try:
    import torch
    from esm.models.esmfold2.processor import ESMFold2InputBuilder, _seed_context
    from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
except ModuleNotFoundError:
    torch = None

CHAIN_LETTERS = string.ascii_uppercase
DOCKQ_CSV   = CSV_DIR / "dockq_ost_esmfold2fast_tts_all_samples.csv"
LM_MASK_PCT = 0.1
SEEDS       = list(range(50))
IFACE_DIST  = 8.0          # Angstrom cutoff for interface residue detection
DEVICE      = "cuda:0"
BASE_LT     = 0.3          # rescue set: baseline DockQ below this ...
MAX_GE      = 0.5          # ... and max-over-seeds DockQ at least this

THREE_TO_ONE = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U",
    "PYL":"O","MSE":"M","HYP":"P","TPO":"T","SEP":"S","PTR":"Y","CSO":"C",
}


# ---------------------------------------------------------------------------
# Targets / DockQ loading
# ---------------------------------------------------------------------------

def load_targets() -> dict:
    """assembly_id -> list of dicts {chain1, chain2, antibody, antigen}."""
    out = defaultdict(list)
    with open(TARGETS_CSV) as f:
        for r in csv.DictReader(f):
            out[r["pdb_id"]].append({
                "chain1": r["interface_chain_id_1"],
                "chain2": r["interface_chain_id_2"],
                "antibody": r["antibody_chain"],
                "antigen": r["antigen_chain"],
            })
    return out


def load_dockq(assembly_id: str, c1: str, c2: str):
    """Return (per_seed, baseline).

    per_seed: {seed_int: (best_dockq, chain_mapping_dict)} using the best sample.
    baseline: best DockQ over baseline_seed samples (or None).
    """
    best = {}            # seed -> (dockq, cm)
    baseline = None
    with open(DOCKQ_CSV) as f:
        for row in csv.DictReader(f):
            if row["assembly_id"] != assembly_id:
                continue
            if {row["chain1"], row["chain2"]} != {c1, c2}:
                continue
            try:
                dq = float(row["dockq"])
            except (ValueError, KeyError):
                continue
            if row["seed"] == "baseline_seed":
                baseline = dq if baseline is None else max(baseline, dq)
                continue
            try:
                seed = int(row["seed"])
            except ValueError:
                continue
            if seed not in best or dq > best[seed][0]:
                cm = json.loads(row["chain_mapping"]) if row.get("chain_mapping") else {}
                best[seed] = (dq, cm)
    return best, baseline


def select_rescue_set(targets: dict) -> list:
    """List of (assembly_id, c1, c2, antibody, antigen, baseline, max_dq)."""
    rescue = []
    for assembly_id, ifaces in targets.items():
        for it in ifaces:
            c1, c2 = it["chain1"], it["chain2"]
            per_seed, baseline = load_dockq(assembly_id, c1, c2)
            if baseline is None or not per_seed:
                continue
            max_dq = max(dq for dq, _ in per_seed.values())
            if baseline < BASE_LT and max_dq >= MAX_GE:
                rescue.append((assembly_id, c1, c2, it["antibody"], it["antigen"],
                               baseline, max_dq))
    rescue.sort(key=lambda x: -x[6])
    return rescue


# ---------------------------------------------------------------------------
# Model input + mask reproduction  (generalized from analyze_masks_7yxu.py)
# ---------------------------------------------------------------------------

def build_spi(pdb_base: str):
    """Return (StructurePredictionInput, model_seqs) for <pdb_base>.json."""
    with open(JSON_DIR / f"{pdb_base}.json") as f:
        data = json.load(f)
    entry = data[0]
    chain_idx = 0
    inputs, model_seqs = [], []
    for seq_entry in entry["sequences"]:
        if "proteinChain" not in seq_entry:
            continue
        chain_data = seq_entry["proteinChain"]
        sequence = chain_data["sequence"]
        count = chain_data.get("count", 1)
        ids = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
        chain_idx += count
        inputs.append(ProteinInput(id=ids if count > 1 else ids[0], sequence=sequence))
        model_seqs.extend([sequence] * count)  # one entry per model chain (asym order)
    return StructurePredictionInput(sequences=inputs), model_seqs


def build_lm_input_ids(features: dict):
    """Reproduce lm_input_ids BOS/EOS construction from compute_lm_hidden_states.

    Returns (lm_input_ids, expand_maps) where expand_maps[b][orig_tok_idx] = lm_pos.
    Verbatim from analyze_masks_7yxu.py (model-agnostic).
    """
    input_ids = features["input_ids"]
    asym_id   = features["asym_id"]
    res_idx   = features["residue_index"]
    mol_type  = features["mol_type"]
    tok_mask  = features["token_attention_mask"]

    B, L = input_ids.shape
    device = input_ids.device
    protein_mask = (mol_type == 0) & tok_mask.bool()

    lm_input_list, lm_lengths, expand_maps = [], [], []
    for b in range(B):
        mask_b = protein_mask[b]
        ids_b  = input_ids[b][mask_b]
        asym_b = asym_id[b][mask_b]
        res_b  = res_idx[b][mask_b]

        keys = torch.stack((asym_b, res_b), dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        n_unique = unique_keys.size(0)
        token_positions = torch.arange(keys.size(0), device=device, dtype=torch.long)
        first_pos = torch.full((n_unique,), keys.size(0), device=device, dtype=torch.long)
        first_pos.scatter_reduce_(0, inverse, token_positions, reduce="amin", include_self=True)
        ordered           = torch.argsort(first_pos)
        first_pos_ordered = first_pos[ordered]
        ids_collapsed     = ids_b[first_pos_ordered]
        asym_collapsed    = asym_b[first_pos_ordered]

        remap = torch.empty_like(ordered)
        remap[ordered] = torch.arange(n_unique, device=device, dtype=torch.long)
        inverse_ordered = remap[inverse]

        chain_ids = asym_collapsed.unique(sorted=True)
        parts = [torch.tensor([0], device=device, dtype=ids_b.dtype)]
        per_token_lm_pos = torch.empty(n_unique, device=device, dtype=torch.long)
        cursor = 1
        for i, cid in enumerate(chain_ids):
            in_chain = (asym_collapsed == cid).nonzero(as_tuple=True)[0]
            parts.append(ids_collapsed[in_chain])
            per_token_lm_pos[in_chain] = torch.arange(
                cursor, cursor + in_chain.shape[0], device=device, dtype=torch.long
            )
            cursor += in_chain.shape[0]
            if i < len(chain_ids) - 1:
                parts.append(torch.tensor([2, 0], device=device, dtype=ids_b.dtype))
                cursor += 2
        parts.append(torch.tensor([2], device=device, dtype=ids_b.dtype))
        lm_seq = torch.cat(parts)
        lm_input_list.append(lm_seq)
        lm_lengths.append(lm_seq.shape[0])

        prot_pos_b = mask_b.nonzero(as_tuple=True)[0]
        expand_map = torch.full((L,), -1, device=device, dtype=torch.long)
        expand_map[prot_pos_b] = per_token_lm_pos[inverse_ordered]
        expand_maps.append(expand_map)

    max_len = max(lm_lengths)
    lm_input_ids = torch.full((B, max_len), 1, device=device, dtype=input_ids.dtype)
    for b in range(B):
        lm_input_ids[b, : lm_lengths[b]] = lm_input_list[b]
    return lm_input_ids, expand_maps


def generate_masks(lm_input_ids, features, expand_maps, seeds=None) -> dict:
    """Return {seed: set of (asym_idx, residue_index) masked residues} (batch 0).

    seeds: iterable of seed ints to reproduce; defaults to module-level SEEDS.
    """
    if seeds is None:
        seeds = SEEDS
    device   = lm_input_ids.device
    asym_id  = features["asym_id"][0]
    res_idx  = features["residue_index"][0]
    tok_mask = features["token_attention_mask"][0]
    mol_type = features["mol_type"][0]
    protein_mask = (mol_type == 0) & tok_mask.bool()
    expand_map   = expand_maps[0]

    seed_to_masked = {}
    for seed in seeds:
        with _seed_context(seed):
            special = (lm_input_ids == 0) | (lm_input_ids == 1) | (lm_input_ids == 2)
            do_mask = (torch.rand(lm_input_ids.shape, device=device) < LM_MASK_PCT) & ~special
        masked_lm_set = set(do_mask[0].nonzero(as_tuple=True)[0].cpu().tolist())
        masked = set()
        for tok_pos in protein_mask.nonzero(as_tuple=True)[0].cpu().tolist():
            lm_pos = expand_map[tok_pos].item()
            if lm_pos >= 0 and lm_pos in masked_lm_set:
                masked.add((asym_id[tok_pos].item(), res_idx[tok_pos].item()))
        seed_to_masked[seed] = masked
    return seed_to_masked


def all_residues(features: dict) -> set:
    tok_mask = features["token_attention_mask"][0]
    mol_type = features["mol_type"][0]
    asym_id  = features["asym_id"][0]
    res_idx  = features["residue_index"][0]
    protein_mask = (mol_type == 0) & tok_mask.bool()
    return {(asym_id[p].item(), res_idx[p].item())
            for p in protein_mask.nonzero(as_tuple=True)[0].cpu().tolist()}


# ---------------------------------------------------------------------------
# Interface residues from ground truth (reference-chain space)
# ---------------------------------------------------------------------------

def get_interface_ref(gt_cif: Path, model_seqs: list, c1: str, c2: str,
                      cutoff: float = IFACE_DIST):
    """Return (side1, side2): sets of (ref_chain_label, model_pos0based).

    side1 lives on ref chain c1, side2 on c2. Residue positions are mapped to the
    MODEL chain's 0-based index by aligning the GT chain sequence to the best-matching
    model sequence (works for homomers since copies share the sequence).
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.sequence as bseq
    import biotite.sequence.align as align

    cif = pdbx.CIFFile.read(gt_cif)
    gt = pdbx.get_structure(cif, model=1, use_author_fields=False)
    submat = align.SubstitutionMatrix.std_protein_matrix()

    def chain_seq(chain_label):
        chain_s = gt[gt.chain_id == chain_label]
        rids, aas = [], []
        for res in struc.residue_iter(chain_s):
            aa = THREE_TO_ONE.get(res[0].res_name)
            if aa is not None:
                rids.append(res[0].res_id)
                aas.append(aa)
        return rids, "".join(aas)

    def res_to_modelpos(chain_label):
        """{gt_res_id -> model_pos0based} via best-matching model seq."""
        rids, seqstr = chain_seq(chain_label)
        if not seqstr:
            return {}
        gt_obj = bseq.ProteinSequence(seqstr)
        best, best_score = None, None
        for ms in set(model_seqs):
            try:
                al = align.align_optimal(gt_obj, bseq.ProteinSequence(ms), submat,
                                         gap_penalty=(-10, -1), terminal_penalty=False)[0]
            except Exception:
                continue
            if best_score is None or al.score > best_score:
                best_score, best = al.score, al.trace
        out = {}
        if best is not None:
            for gt_pos, mod_pos in best:
                if gt_pos >= 0 and mod_pos >= 0:
                    out[rids[gt_pos]] = mod_pos
        return out

    map1 = res_to_modelpos(c1)
    map2 = res_to_modelpos(c2)

    s1 = gt[(gt.chain_id == c1) & ~struc.filter_solvent(gt)]
    s2 = gt[(gt.chain_id == c2) & ~struc.filter_solvent(gt)]
    if len(s1) == 0 or len(s2) == 0:
        return set(), set()
    cell = struc.CellList(s2, cell_size=cutoff)
    contacts1, contacts2 = set(), set()
    for atom in s1:
        near = cell.get_atoms(atom.coord, radius=cutoff)
        if len(near) > 0:
            contacts1.add(atom.res_id)
            for idx in near:
                contacts2.add(s2[idx].res_id)
    side1 = {(c1, map1[r]) for r in contacts1 if r in map1}
    side2 = {(c2, map2[r]) for r in contacts2 if r in map2}
    return side1, side2


# ---------------------------------------------------------------------------
# Per-interface analysis
# ---------------------------------------------------------------------------

def cm_asym(cm: dict, ref_label: str):
    """Reference chain label -> model asym_idx via chain_mapping; None if absent."""
    model_letter = cm.get(ref_label)
    if not model_letter or len(model_letter) != 1:
        return None
    return ord(model_letter) - ord("A")


def analyze_interface(builder, assembly_id, c1, c2, antibody, antigen, gt_cif):
    """Return per_seed feature rows + per-residue records, or None on failure."""
    pdb_base = assembly_id.split("-")[0]
    spi, model_seqs = build_spi(pdb_base)
    features, _ = builder.prepare_input(spi, seed=0, device=DEVICE)
    lm_input_ids, expand_maps = build_lm_input_ids(features)
    seed_to_masked = generate_masks(lm_input_ids, features, expand_maps)
    n_total = len(all_residues(features))

    side1, side2 = get_interface_ref(gt_cif, model_seqs, c1, c2)
    by_label = {c1: side1, c2: side2}
    epitope_ref  = by_label.get(antigen, set())   # antigen side
    paratope_ref = by_label.get(antibody, set())  # antibody side
    iface_ref    = epitope_ref | paratope_ref
    if not iface_ref:
        return None

    per_seed, baseline = load_dockq(assembly_id, c1, c2)

    def masked_in(seed, cm, ref_set):
        n = 0
        for ref_label, pos in ref_set:
            a = cm_asym(cm, ref_label)
            if a is not None and (a, pos) in seed_to_masked[seed]:
                n += 1
        return n

    rows = []
    # per-residue: (ref_label, pos, region) -> [(masked_bool, dockq), ...]
    res_obs = defaultdict(list)
    region_of = {}
    for r in epitope_ref:  region_of[r] = "epitope"
    for r in paratope_ref: region_of[r] = "paratope"

    for seed in SEEDS:
        if seed not in per_seed:
            continue
        dq, cm = per_seed[seed]
        n_iface = masked_in(seed, cm, iface_ref)
        n_epi   = masked_in(seed, cm, epitope_ref)
        n_para  = masked_in(seed, cm, paratope_ref)
        n_mask_total = len(seed_to_masked[seed])
        rows.append({
            "assembly_id": assembly_id, "chain1": c1, "chain2": c2, "seed": seed,
            "dockq": dq, "baseline": baseline,
            "frac_total_masked": n_mask_total / max(n_total, 1),
            "frac_iface_masked": n_iface / max(len(iface_ref), 1),
            "frac_epitope_masked": n_epi / max(len(epitope_ref), 1) if epitope_ref else float("nan"),
            "frac_paratope_masked": n_para / max(len(paratope_ref), 1) if paratope_ref else float("nan"),
            "n_iface_res": len(iface_ref),
            "n_epitope_res": len(epitope_ref),
            "n_paratope_res": len(paratope_ref),
        })
        for ref_label, pos in iface_ref:
            a = cm_asym(cm, ref_label)
            masked = a is not None and (a, pos) in seed_to_masked[seed]
            res_obs[(ref_label, pos, region_of[(ref_label, pos)])].append((masked, dq))

    # per-residue sensitivity
    res_rows = []
    for (ref_label, pos, region), obs in res_obs.items():
        m = [dq for masked, dq in obs if masked]
        u = [dq for masked, dq in obs if not masked]
        if not m or not u:
            continue
        res_rows.append({
            "assembly_id": assembly_id, "chain1": c1, "chain2": c2,
            "ref_chain": ref_label, "pos": pos, "region": region,
            "n_masked": len(m),
            "sensitivity": float(np.mean(m) - np.mean(u)),
        })
    return rows, res_rows


# ---------------------------------------------------------------------------
# Pooled stats + plots
# ---------------------------------------------------------------------------

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.all(x == x[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def sign_test_p(diffs):
    """Two-sided sign test p-value for a list of paired differences."""
    d = [x for x in diffs if x == x]
    n = len(d)
    if n == 0:
        return float("nan"), 0, 0
    neg = sum(1 for x in d if x < 0)
    k = min(neg, n - neg)
    p = min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)
    return p, neg, n


def success_contrast(by_iface, feature, thr):
    """Paired per-interface mean(feature | success) − mean(feature | fail) where
    success = best-sample DockQ >= thr. Returns (mean_delta, neg, n, sign_p)."""
    diffs = []
    for rows in by_iface.values():
        succ = [r for r in rows if r["dockq"] >= thr]
        fail = [r for r in rows if r["dockq"] < thr]
        if len(succ) < 2 or len(fail) < 2:
            continue
        sv = [r[feature] for r in succ if r[feature] == r[feature]]
        fv = [r[feature] for r in fail if r[feature] == r[feature]]
        if sv and fv:
            diffs.append(float(np.mean(sv) - np.mean(fv)))
    p, neg, n = sign_test_p(diffs)
    return (float(np.mean(diffs)) if diffs else float("nan")), neg, n, p


CONTRAST_FEATURES = [
    ("frac_iface_masked",    "interface"),
    ("frac_epitope_masked",  "epitope"),
    ("frac_paratope_masked", "paratope"),
    ("frac_total_masked",    "TOTAL (control)"),
    ("iface_rel",            "interface − total"),
    ("epi_rel",              "epitope − total"),
    ("para_rel",             "paratope − total"),
]
THRESHOLDS = [0.23, 0.5]


def main():
    summary_only = "--summary-only" in sys.argv
    print("Loading targets + selecting rescue set...")
    targets = load_targets()
    rescue = select_rescue_set(targets)
    print(f"Rescue set: {len(rescue)} interfaces (baseline<{BASE_LT}, max>={MAX_GE})")

    if summary_only:
        feature_rows = _load_csv(CSV_DIR / "mask_features_per_seed.csv")
        res_rows = _load_csv(CSV_DIR / "mask_sensitivity_per_residue.csv")
        print(f"Loaded {len(feature_rows)} feature rows, {len(res_rows)} residue rows from disk")
    else:
        builder = ESMFold2InputBuilder()
        feature_rows, res_rows = [], []
        for assembly_id, c1, c2, antibody, antigen, baseline, max_dq in rescue:
            gt_cif = GROUND_TRUTHS / f"{assembly_id}.cif"
            if not gt_cif.exists():
                print(f"  SKIP {assembly_id} {c1}|{c2}: no ground truth")
                continue
            try:
                res = analyze_interface(builder, assembly_id, c1, c2, antibody, antigen, gt_cif)
            except Exception as e:
                print(f"  ERROR {assembly_id} {c1}|{c2}: {e}")
                continue
            if res is None:
                print(f"  SKIP {assembly_id} {c1}|{c2}: no interface residues mapped")
                continue
            rows, rrows = res
            feature_rows.extend(rows)
            res_rows.extend(rrows)
            print(f"  {assembly_id} {c1}|{c2}: base={baseline:.3f} max={max_dq:.3f} "
                  f"iface_res={rows[0]['n_iface_res']}")
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        _write_csv(CSV_DIR / "mask_features_per_seed.csv", feature_rows)
        _write_csv(CSV_DIR / "mask_sensitivity_per_residue.csv", res_rows)
        print(f"\nWrote {len(feature_rows)} feature rows, {len(res_rows)} residue rows")

    # derived background-relative columns (region masking minus global rate)
    for r in feature_rows:
        r["iface_rel"] = r["frac_iface_masked"] - r["frac_total_masked"]
        r["epi_rel"] = r["frac_epitope_masked"] - r["frac_total_masked"]
        r["para_rel"] = r["frac_paratope_masked"] - r["frac_total_masked"]

    summarize_and_plot(feature_rows, res_rows, rescue)


def _load_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k in ("assembly_id", "chain1", "chain2", "ref_chain", "region"):
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                r[k] = float("nan")
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path}")


def summarize_and_plot(feature_rows, res_rows, rescue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_iface = defaultdict(list)
    for r in feature_rows:
        by_iface[(r["assembly_id"], r["chain1"], r["chain2"])].append(r)

    # --- Main test: success vs fail seed contrast, with total-masking control ---
    print("\n=== Rescued vs not: mask features (paired per-interface, sign test) ===")
    contrast_results = {}
    for thr in THRESHOLDS:
        print(f"  success = best-sample DockQ >= {thr}")
        for feat, label in CONTRAST_FEATURES:
            m, neg, n, p = success_contrast(by_iface, feat, thr)
            contrast_results[(thr, feat)] = (m, neg, n, p)
            star = " *" if p < 0.05 else ""
            print(f"    {label:<18} Δ(succ−fail)={m:+.4f}  neg {neg}/{n}  sign-p={p:.3f}{star}")

    # --- Per-residue sensitivity by region (pooled) ---
    by_region = defaultdict(list)
    for r in res_rows:
        by_region[r["region"]].append(r["sensitivity"])
    print("\n=== Per-residue sensitivity by region (pooled) ===")
    for region, vals in sorted(by_region.items()):
        print(f"  {region:>9}: mean {np.mean(vals):+.4f} "
              f"(n={len(vals)}, {100*sum(1 for v in vals if v<0)/len(vals):.0f}% negative)")

    _plot_contrast(contrast_results, plt)
    _plot_region_box(by_region, plt)
    _plot_scatter(by_iface, plt)
    write_report(by_iface, rescue, contrast_results, by_region)


def _plot_contrast(contrast_results, plt):
    """Bar of Δ(success−fail) per feature at the strict threshold; control highlighted."""
    thr = 0.5
    feats = [f for f, _ in CONTRAST_FEATURES]
    labels = [l for _, l in CONTRAST_FEATURES]
    vals = [contrast_results[(thr, f)][0] for f in feats]
    ps = [contrast_results[(thr, f)][3] for f in feats]
    colors = ["gray" if "control" in l else ("steelblue" if p < 0.05 else "lightsteelblue")
              for l, p in zip(labels, ps)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(feats)), vals, color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    for i, (v, p) in enumerate(zip(vals, ps)):
        # anchor labels just inside the zero line so they never hit the y-axis labels
        ax.text(-0.0003, i, f"p={p:.3f}", va="center", ha="right", fontsize=8)
    ax.set_xlabel("Δ mask fraction  (rescued seeds − failed seeds),  DockQ≥0.5")
    ax.set_title("Rescued seeds mask the interface less — beyond the total-masking control\n"
                 "(negative = rescued seeds masked that region less)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mask_rescue_contrast.png", dpi=150)
    print(f"\n  saved {RESULTS_DIR / 'mask_rescue_contrast.png'}")


def _plot_region_box(by_region, plt):
    fig, ax = plt.subplots(figsize=(7, 5))
    regions = ["epitope", "paratope"]
    data = [by_region[r] for r in regions if by_region[r]]
    labels = [r for r in regions if by_region[r]]
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel("per-residue sensitivity  (mean DockQ masked − unmasked)")
        ax.set_title("Per-residue effect of masking interface residues (pooled)\n"
                     "below 0 = masking that residue hurts DockQ")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "mask_rescue_region_effect.png", dpi=150)
        print(f"  saved {RESULTS_DIR / 'mask_rescue_region_effect.png'}")


def _plot_scatter(by_iface, plt):
    n = len(by_iface)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (key, rows) in zip(axes.flat,
                               sorted(by_iface.items(), key=lambda kv: -max(r["dockq"] for r in kv[1]))):
        ax.axis("on")
        ax.scatter([r["frac_iface_masked"] for r in rows], [r["dockq"] for r in rows],
                   s=18, alpha=0.6, color="steelblue")
        ax.axhline(rows[0]["baseline"], color="darkorange", linestyle="--", linewidth=1)
        ax.set_title(f"{key[0].split('-')[0]} {key[1]}|{key[2]}", fontsize=9)
        ax.set_xlabel("frac iface masked", fontsize=8)
        ax.set_ylabel("DockQ", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Rescue set: fraction of interface masked vs DockQ (orange = no-mask baseline)", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mask_rescue_scatter.png", dpi=150, bbox_inches="tight")
    print(f"  saved {RESULTS_DIR / 'mask_rescue_scatter.png'}")


def write_report(by_iface, rescue, contrast_results, by_region):
    out = RESULTS_DIR / "markdown_docs" / "why_masks_rescue.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    base_max = {(a, c1, c2): (b, m) for a, c1, c2, _, _, b, m in rescue}
    L = []
    L.append("# Why do certain input masks rescue hard antibody–antigen interfaces?\n")
    L.append(f"Rescue set: **{len(by_iface)} interfaces** with baseline DockQ < {BASE_LT} and "
             f"max-over-seeds DockQ ≥ {MAX_GE}. ESMFold2-Fast, 50 TTS seeds "
             f"(`lm_mask_pct={LM_MASK_PCT}`); masks reproduced deterministically per seed. "
             "Per-seed DockQ = best of the 5 diffusion samples.\n")
    L.append("**Hypothesis:** a good mask hits non-interface distractor regions more than the "
             "interface, forcing the network onto the true interface.\n")
    L.append("## Headline\n")
    iface_m, iface_neg, iface_n, iface_p = contrast_results[(0.5, "frac_iface_masked")]
    tot_m, _, _, tot_p = contrast_results[(0.5, "frac_total_masked")]
    rel_m, rel_neg, rel_n, rel_p = contrast_results[(0.5, "iface_rel")]
    L.append(f"Seeds that genuinely rescue an interface (DockQ ≥ 0.5) mask the interface "
             f"**less** than the seeds that fail: Δ = {iface_m:+.4f} ({iface_neg}/{iface_n} "
             f"interfaces, sign-p = {iface_p:.3f}). Those seeds also mask **less overall** "
             f"(total Δ = {tot_m:+.4f}, p = {tot_p:.3f}), but the interface is spared **beyond** "
             f"that global reduction: interface-minus-total Δ = {rel_m:+.4f} "
             f"({rel_neg}/{rel_n}, p = {rel_p:.3f}). The effect is clear for big rescues "
             f"(DockQ ≥ 0.5) and weak at the marginal threshold (0.23).\n")
    L.append("## Rescued vs failed seeds — mask fraction by region\n")
    L.append("Paired per-interface Δ = mean(feature | rescued) − mean(feature | failed); "
             "sign test across interfaces. `*` = total-masking control. "
             "`− total` rows subtract the global masking rate (the confound-controlled test).\n")
    for thr in THRESHOLDS:
        L.append(f"**success = DockQ ≥ {thr}**\n")
        L.append("| feature | Δ (rescued − failed) | neg/total | sign-p |")
        L.append("|---|---|---|---|")
        for feat, label in CONTRAST_FEATURES:
            m, neg, n, p = contrast_results[(thr, feat)]
            L.append(f"| {label} | {m:+.4f} | {neg}/{n} | {p:.3f} |")
        L.append("")
    L.append("## Per-residue sensitivity by region (pooled)\n")
    L.append("Sensitivity = mean(DockQ | residue masked) − mean(DockQ | unmasked); "
             "negative = masking that residue hurts.\n")
    L.append("| region | mean sensitivity | n residues | % negative |")
    L.append("|---|---|---|---|")
    for region, vals in sorted(by_region.items()):
        L.append(f"| {region} | {np.mean(vals):+.4f} | {len(vals)} | "
                 f"{100*sum(1 for v in vals if v<0)/len(vals):.0f}% |")
    L.append("\n## Per-interface ρ(frac_iface_masked, DockQ)\n")
    L.append("| interface | baseline | max DockQ | ρ |")
    L.append("|---|---|---|---|")
    rows_rho = []
    for key, rows in by_iface.items():
        rho = spearman([r["frac_iface_masked"] for r in rows], [r["dockq"] for r in rows])
        rows_rho.append((key, rho))
    for key, rho in sorted(rows_rho, key=lambda t: (t[1] if t[1] == t[1] else 0)):
        b, m = base_max.get(key, (float("nan"), float("nan")))
        L.append(f"| {key[0]} {key[1]}\\|{key[2]} | {b:.3f} | {m:.3f} | {rho:+.3f} |")
    L.append("\n**Caveat:** 7yxu is the cleanest single exemplar; pooled, the effect is real in "
             "direction and survives the total-masking control, but is modest and threshold-"
             "dependent. ~Half the rescue set is homomeric (per-sample chain remapping adds noise).\n")
    L.append("Artifacts: `mask_rescue_contrast.png`, `mask_rescue_region_effect.png`, "
             "`mask_rescue_scatter.png`, `csv_files/mask_features_per_seed.csv`, "
             "`csv_files/mask_sensitivity_per_residue.csv`.\n")
    out.write_text("\n".join(L))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
