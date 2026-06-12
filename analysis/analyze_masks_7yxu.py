"""
Analyze whether masks for good-DockQ seeds share patterns for 7yxu-assembly1.

Steps:
1. Build lm_input_ids for 7yxu using ESMFold2InputBuilder (no model weights needed)
2. For each seed 0-49, reproduce the exact CUDA mask via _seed_context
3. Find interface residues (A|B and A|C) from ground truth CIF via 8A distance cutoff
   (sequence-aligns GT chains to model chains to map positions)
4. Compute per-seed stats: frac interface masked for A|B and A|C, total masked
5. Hierarchical clustering (Hamming distance) on mask binary vectors
6. k-means (k=2) to check good/bad split

Usage (needs the esm venv, which has `esm` installed):
    ~/esm/.venv/bin/python analysis/analyze_masks_7yxu.py
"""
import csv
import json
import string
import sys
from pathlib import Path

import numpy as np
import torch

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import JSON_DIR, GROUND_TRUTHS, ESM_PREDICTIONS, RESULTS_DIR

# Requires the esm venv (esm installed editable): ~/esm/.venv/bin/python
from esm.models.esmfold2.processor import ESMFold2InputBuilder, _seed_context
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput

CHAIN_LETTERS = string.ascii_uppercase
JSON_PATH  = JSON_DIR / "7yxu.json"
GT_CIF     = GROUND_TRUTHS / "7yxu-assembly1.cif"
DOCKQ_CSV  = ESM_PREDICTIONS / "dockq_ost_esmfold2fast_tts_all_samples.csv"
LM_MASK_PCT = 0.1
SEEDS       = list(range(50))
IFACE_DIST  = 8.0  # Angstrom cutoff for interface residue detection
DEVICE      = "cuda:0"

GOOD_SEEDS = {14, 45, 47, 49}


# ---------------------------------------------------------------------------
# 1. Build model input (no model weights needed)
# ---------------------------------------------------------------------------

def build_spi() -> StructurePredictionInput:
    with open(JSON_PATH) as f:
        data = json.load(f)
    entry = data[0]
    chain_idx = 0
    inputs = []
    for seq_entry in entry["sequences"]:
        if "proteinChain" not in seq_entry:
            continue
        chain_data = seq_entry["proteinChain"]
        sequence   = chain_data["sequence"]
        count      = chain_data.get("count", 1)
        ids        = [CHAIN_LETTERS[chain_idx + i] for i in range(count)]
        chain_idx += count
        inputs.append(ProteinInput(id=ids if count > 1 else ids[0], sequence=sequence))
    return StructurePredictionInput(sequences=inputs)


def build_lm_input_ids(features: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Reproduce the lm_input_ids BOS/EOS construction from compute_lm_hidden_states.

    Returns (lm_input_ids, expand_maps) where expand_maps[b][orig_tok_idx] = lm_pos.
    """
    input_ids  = features["input_ids"]
    asym_id    = features["asym_id"]
    res_idx    = features["residue_index"]
    mol_type   = features["mol_type"]
    tok_mask   = features["token_attention_mask"]

    B, L   = input_ids.shape
    device = input_ids.device
    protein_mask = (mol_type == 0) & tok_mask.bool()

    lm_input_list = []
    lm_lengths    = []
    expand_maps   = []

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
        parts: list[torch.Tensor] = [torch.tensor([0], device=device, dtype=ids_b.dtype)]
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


# ---------------------------------------------------------------------------
# 2. Generate masks per seed (exact CUDA reproduction)
# ---------------------------------------------------------------------------

def generate_masks(lm_input_ids: torch.Tensor, features: dict, expand_maps: list) -> dict[int, set]:
    """Return {seed: set of (asym_idx, residue_index) masked residues}."""
    B, max_len = lm_input_ids.shape
    device     = lm_input_ids.device

    asym_id   = features["asym_id"][0]
    res_idx   = features["residue_index"][0]
    tok_mask  = features["token_attention_mask"][0]
    mol_type  = features["mol_type"][0]
    protein_mask = (mol_type == 0) & tok_mask.bool()
    expand_map   = expand_maps[0]  # batch size 1

    seed_to_masked = {}
    for seed in SEEDS:
        with _seed_context(seed):
            special = (
                (lm_input_ids == 0) | (lm_input_ids == 1) | (lm_input_ids == 2)
            )
            do_mask = (
                torch.rand(lm_input_ids.shape, device=device) < LM_MASK_PCT
            ) & ~special

        masked_lm_positions = do_mask[0].nonzero(as_tuple=True)[0]
        masked_lm_set = set(masked_lm_positions.cpu().tolist())

        prot_pos = protein_mask.nonzero(as_tuple=True)[0]
        masked_tokens = set()
        for tok_pos in prot_pos.cpu().tolist():
            lm_pos = expand_map[tok_pos].item()
            if lm_pos >= 0 and lm_pos in masked_lm_set:
                asym = asym_id[tok_pos].item()
                res  = res_idx[tok_pos].item()
                masked_tokens.add((asym, res))

        seed_to_masked[seed] = masked_tokens

    return seed_to_masked


# ---------------------------------------------------------------------------
# 3. Interface residues from ground truth CIF
# ---------------------------------------------------------------------------

def get_interface_residues(cutoff: float = IFACE_DIST) -> tuple[set, set]:
    """Find interface residues (model coords) for A|B and A|C using 8A cutoff.

    Returns (iface_AB, iface_AC) as sets of (asym_idx, res_idx_0based).
    The model uses 0-based sequential residue_index per chain.
    We align GT sequences to model sequences to map positions.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.sequence as bseq
    import biotite.sequence.align as align

    # Load GT structure with label chain names
    cif_file = pdbx.CIFFile.read(GT_CIF)
    gt_struct = pdbx.get_structure(cif_file, model=1, use_author_fields=False)

    # Load model sequences from JSON
    with open(JSON_PATH) as f:
        data = json.load(f)
    entry = data[0]
    model_seqs = []
    for seq_entry in entry["sequences"]:
        if "proteinChain" in seq_entry:
            model_seqs.append(seq_entry["proteinChain"]["sequence"])

    gt_chains = ["A", "B", "C"]  # label chains that correspond to model chains A,B,C

    # Build mapping: (gt_chain, gt_res_id) -> (model_asym_idx, model_res_idx_0based)
    substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()

    gt_res_to_model = {}  # (gt_chain_label, gt_res_id) -> (model_asym_idx, model_res_0based)
    THREE_TO_ONE = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U",
        "PYL":"O","MSE":"M","HYP":"P","TPO":"T","SEP":"S","PTR":"Y","CSO":"C",
    }

    for ci, (gt_chain, model_seq) in enumerate(zip(gt_chains, model_seqs)):
        chain_s = gt_struct[gt_struct.chain_id == gt_chain]
        gt_res_ids = []
        gt_aa_list = []
        for res in struc.residue_iter(chain_s):
            rid     = res[0].res_id
            resname = res[0].res_name
            aa = THREE_TO_ONE.get(resname, None)
            if aa is not None:
                gt_res_ids.append(rid)
                gt_aa_list.append(aa)

        gt_seq_str  = "".join(gt_aa_list)
        gt_seq_obj  = bseq.ProteinSequence(gt_seq_str)
        mod_seq_obj = bseq.ProteinSequence(model_seq)

        alignment = align.align_optimal(
            gt_seq_obj, mod_seq_obj,
            substitution_matrix,
            gap_penalty=(-10, -1),
            terminal_penalty=False,
        )[0]

        trace = alignment.trace  # shape (n_aligned, 2): [gt_pos, model_pos], -1 = gap
        for gt_pos, mod_pos in trace:
            if gt_pos >= 0 and mod_pos >= 0:
                gt_res_id = gt_res_ids[gt_pos]
                gt_res_to_model[(gt_chain, gt_res_id)] = (ci, mod_pos)

    # Find interface contacts by distance
    def get_interface(chain1_label: str, chain2_label: str) -> tuple[set, set]:
        s1 = gt_struct[(gt_struct.chain_id == chain1_label) & ~struc.filter_solvent(gt_struct)]
        s2 = gt_struct[(gt_struct.chain_id == chain2_label) & ~struc.filter_solvent(gt_struct)]
        if len(s1) == 0 or len(s2) == 0:
            return set(), set()
        cell = struc.CellList(s2, cell_size=cutoff)
        contacts1 = set()
        contacts2 = set()
        for atom in s1:
            near = cell.get_atoms(atom.coord, radius=cutoff)
            if len(near) > 0:
                contacts1.add(atom.res_id)
                for idx in near:
                    contacts2.add(s2[idx].res_id)
        iface1 = set()
        for rid in contacts1:
            key = (chain1_label, rid)
            if key in gt_res_to_model:
                iface1.add(gt_res_to_model[key])
        iface2 = set()
        for rid in contacts2:
            key = (chain2_label, rid)
            if key in gt_res_to_model:
                iface2.add(gt_res_to_model[key])
        return iface1 | iface2, iface1, iface2

    iface_AB_all, iface_A_in_AB, iface_B_in_AB = get_interface("A", "B")
    iface_AC_all, iface_A_in_AC, iface_C_in_AC = get_interface("A", "C")

    print(f"Interface A|B: {len(iface_AB_all)} model residues "
          f"(A: {len(iface_A_in_AB)}, B: {len(iface_B_in_AB)})")
    print(f"Interface A|C: {len(iface_AC_all)} model residues "
          f"(A: {len(iface_A_in_AC)}, C: {len(iface_C_in_AC)})")

    return iface_AB_all, iface_AC_all


# ---------------------------------------------------------------------------
# 4. Load DockQ per seed (best sample per seed)
# ---------------------------------------------------------------------------

def load_dockq_per_seed() -> tuple[dict[int, float], dict[int, float]]:
    """Returns {seed: best_dockq} for A|B and A|C interfaces."""
    ab: dict[int, float] = {}
    ac: dict[int, float] = {}
    with open(DOCKQ_CSV) as f:
        for row in csv.DictReader(f):
            if row["assembly_id"] != "7yxu-assembly1":
                continue
            try:
                seed = int(row["seed"])
                dq   = float(row["dockq"])
            except (ValueError, KeyError):
                continue
            c1, c2 = row["chain1"], row["chain2"]
            if {c1, c2} == {"A", "B"}:
                ab[seed] = max(ab.get(seed, 0.0), dq)
            elif {c1, c2} == {"A", "C"}:
                ac[seed] = max(ac.get(seed, 0.0), dq)
    return ab, ac


# ---------------------------------------------------------------------------
# 5. Analysis
# ---------------------------------------------------------------------------

def get_cdr_residues() -> dict[str, set]:
    """Return model (asym_idx, res_idx_0based) sets for each CDR.

    CDR positions identified from the 7yxu sequences via conserved anchor residues:
      CDR-H3: chain B positions 98-109  (SSYYGSYYSIDY, bounded by C..CAR and WGQG)
      CDR-L3: chain C positions 88-96   (QQYYDNLPT, bounded by C and FGQG)
      CDR-H1: chain B positions 26-31   (GFTFSY, Chothia 26-32)
      CDR-H2: chain B positions 52-57   (SSGSTS, Chothia 52-56)
      CDR-L1: chain C positions 24-33   (RASQSISSYL, Chothia 24-34)
      CDR-L2: chain C positions 50-55   (AASSLQ, Chothia 50-56)
    """
    # asym_idx: A=0, B=1, C=2
    H, L = 1, 2
    return {
        "CDR-H1": {(H, r) for r in range(26, 32)},
        "CDR-H2": {(H, r) for r in range(52, 58)},
        "CDR-H3": {(H, r) for r in range(98, 110)},
        "CDR-L1": {(L, r) for r in range(24, 34)},
        "CDR-L2": {(L, r) for r in range(50, 56)},
        "CDR-L3": {(L, r) for r in range(88, 97)},
    }


def compute_stats(
    seed_to_masked: dict[int, set],
    iface_AB: set,
    iface_AC: set,
    cdr_residues: dict[str, set],
    features: dict,
) -> dict[int, dict]:
    """Compute per-seed stats."""
    tok_mask = features["token_attention_mask"][0]
    mol_type = features["mol_type"][0]
    asym_id  = features["asym_id"][0]
    res_idx  = features["residue_index"][0]

    protein_mask = (mol_type == 0) & tok_mask.bool()

    all_residues = set()
    for pos in protein_mask.nonzero(as_tuple=True)[0].cpu().tolist():
        all_residues.add((asym_id[pos].item(), res_idx[pos].item()))

    results = {}
    for seed in SEEDS:
        masked = seed_to_masked[seed]
        n_total  = len(all_residues)
        n_masked = len(masked)

        row = {
            "n_masked": n_masked,
            "frac_total_masked":   n_masked / max(n_total, 1),
            "frac_iface_AB_masked": len(masked & iface_AB) / max(len(iface_AB), 1),
            "frac_iface_AC_masked": len(masked & iface_AC) / max(len(iface_AC), 1),
        }
        for cdr_name, cdr_set in cdr_residues.items():
            row[f"frac_{cdr_name}_masked"] = len(masked & cdr_set) / max(len(cdr_set), 1)

        results[seed] = row
    return results


def build_mask_matrix(seed_to_masked: dict[int, set], features: dict) -> np.ndarray:
    """Build (n_seeds x n_residues) binary matrix for clustering."""
    tok_mask = features["token_attention_mask"][0]
    mol_type = features["mol_type"][0]
    asym_id  = features["asym_id"][0]
    res_idx  = features["residue_index"][0]

    protein_mask = (mol_type == 0) & tok_mask.bool()
    all_res = sorted(set(
        (asym_id[pos].item(), res_idx[pos].item())
        for pos in protein_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    ))
    res_to_col = {r: i for i, r in enumerate(all_res)}

    mat = np.zeros((len(SEEDS), len(all_res)), dtype=np.uint8)
    for si, seed in enumerate(SEEDS):
        for res in seed_to_masked[seed]:
            if res in res_to_col:
                mat[si, res_to_col[res]] = 1
    return mat


def run_clustering(mat: np.ndarray, dockq_ab: dict[int, float]) -> None:
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist
        from sklearn.cluster import KMeans
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist = pdist(mat, metric="hamming")
        Z = linkage(dist, method="ward")

        fig, ax = plt.subplots(figsize=(14, 5))
        dend = dendrogram(Z, labels=SEEDS, ax=ax, leaf_font_size=8)
        leaves = dend["leaves"]  # order of seeds in dendrogram
        colors = ["red" if SEEDS[i] in GOOD_SEEDS else "black" for i in leaves]
        for lbl, col in zip(ax.get_xticklabels(), colors):
            lbl.set_color(col)
        ax.set_title("7yxu mask clustering (Hamming, Ward) — red = good DockQ seeds (14,45,47,49)")
        ax.set_xlabel("Seed")
        out = RESULTS_DIR / "7yxu_mask_dendrogram.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"\nDendrogram saved to {out}")

        # k-means k=2
        km = KMeans(n_clusters=2, random_state=42, n_init=20)
        labels = km.fit_predict(mat.astype(float))
        cluster_good = {}
        for seed, lbl in zip(SEEDS, labels):
            cluster_good.setdefault(lbl, []).append(seed)
        print("\nk-means k=2 clusters:")
        for k, seeds in cluster_good.items():
            good = [s for s in seeds if s in GOOD_SEEDS]
            print(f"  cluster {k}: {len(seeds)} seeds, good seeds in cluster: {good}")

    except ImportError as e:
        print(f"Clustering skipped (missing dependency): {e}")


def plot_scatter(stats: dict, dockq_ab: dict, dockq_ac: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        # --- Figure 1: frac_masked vs DockQ for each interface (original) ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (iface_key, dockq_dict, title) in zip(axes, [
            ("frac_iface_AB_masked", dockq_ab, "Interface A|B"),
            ("frac_iface_AC_masked", dockq_ac, "Interface A|C"),
        ]):
            for seed in SEEDS:
                x = stats[seed][iface_key]
                y = dockq_dict.get(seed, float("nan"))
                if y != y:
                    continue
                is_good = seed in GOOD_SEEDS
                ax.scatter(x, y,
                           color="red" if is_good else "steelblue",
                           s=60 if is_good else 30,
                           alpha=0.9 if is_good else 0.6,
                           zorder=3 if is_good else 2)
                if is_good:
                    ax.annotate(str(seed), (x, y),
                                textcoords="offset points", xytext=(5, 4),
                                fontsize=8, color="red")
            ax.axhline(0.23, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Fraction of interface residues masked")
            ax.set_ylabel("DockQ")
            ax.set_title(f"7yxu {title}")
            ax.grid(True, alpha=0.3)
            ax.legend(handles=[
                plt.scatter([], [], color="red",       s=60,        label="good seeds"),
                plt.scatter([], [], color="steelblue", s=30, alpha=0.6, label="other seeds"),
                mlines.Line2D([], [], color="gray", linestyle="--", label="DockQ=0.23"),
            ])
        out1 = RESULTS_DIR / "7yxu_mask_scatter.png"
        out1.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out1, dpi=150)
        print(f"Scatter plot saved to {out1}")

        # --- Figure 2: 2D scatter frac_AB vs frac_AC, colored by avg DockQ ---
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        avg_dq = {s: (dockq_ab.get(s, 0) + dockq_ac.get(s, 0)) / 2 for s in SEEDS}
        xs = [stats[s]["frac_iface_AB_masked"] for s in SEEDS]
        ys = [stats[s]["frac_iface_AC_masked"] for s in SEEDS]
        cs = [avg_dq[s] for s in SEEDS]

        sc = ax2.scatter(xs, ys, c=cs, cmap="RdYlGn", s=50, vmin=0, vmax=0.7, zorder=2)
        plt.colorbar(sc, ax=ax2, label="Avg DockQ (A|B + A|C) / 2")

        for seed in GOOD_SEEDS:
            ax2.annotate(str(seed),
                         (stats[seed]["frac_iface_AB_masked"],
                          stats[seed]["frac_iface_AC_masked"]),
                         textcoords="offset points", xytext=(5, 4),
                         fontsize=8, color="darkgreen", fontweight="bold")

        ax2.set_xlabel("Frac A|B interface residues masked")
        ax2.set_ylabel("Frac A|C interface residues masked")
        ax2.set_title("7yxu: both interfaces masked vs avg DockQ")
        ax2.grid(True, alpha=0.3)
        out2 = RESULTS_DIR / "7yxu_mask_scatter_2d.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=150)
        print(f"2D scatter saved to {out2}")

    except Exception as e:
        print(f"Scatter plot failed: {e}")


def plot_cdr_scatter(stats: dict, dockq_ab: dict, dockq_ac: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        avg_dq = {s: (dockq_ab.get(s, 0) + dockq_ac.get(s, 0)) / 2 for s in SEEDS}
        cdrs   = ["CDR-H1", "CDR-H2", "CDR-H3", "CDR-L1", "CDR-L2", "CDR-L3"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()

        for ax, cdr in zip(axes, cdrs):
            for seed in SEEDS:
                x = stats[seed][f"frac_{cdr}_masked"]
                y = avg_dq[seed]
                is_good = seed in GOOD_SEEDS
                ax.scatter(x, y,
                           color="red" if is_good else "steelblue",
                           s=60 if is_good else 25,
                           alpha=0.9 if is_good else 0.55,
                           zorder=3 if is_good else 2)
                if is_good:
                    ax.annotate(str(seed), (x, y),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=7, color="red")
            ax.axhline(0.23, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Fraction masked")
            ax.set_ylabel("Avg DockQ")
            ax.set_title(cdr)
            ax.grid(True, alpha=0.3)

        fig.suptitle("7yxu: CDR masking fraction vs avg DockQ", fontsize=13)
        fig.legend(handles=[
            plt.scatter([], [], color="red",       s=60,        label="good seeds (14,45,47,49)"),
            plt.scatter([], [], color="steelblue", s=25, alpha=0.55, label="other seeds"),
            mlines.Line2D([], [], color="gray", linestyle="--", label="DockQ=0.23"),
        ], loc="lower right")

        out = RESULTS_DIR / "7yxu_cdr_scatter.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"CDR scatter saved to {out}")

    except Exception as e:
        print(f"CDR scatter failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building model input...")
    spi = build_spi()
    builder = ESMFold2InputBuilder()
    features, _ = builder.prepare_input(spi, seed=0, device=DEVICE)
    features = {k: v for k, v in features.items()}  # keep on CUDA

    print("Building lm_input_ids...")
    lm_input_ids, expand_maps = build_lm_input_ids(features)
    print(f"  lm_input_ids shape: {lm_input_ids.shape}")

    print("Generating masks for all seeds...")
    seed_to_masked = generate_masks(lm_input_ids, features, expand_maps)

    print("Identifying interface residues from ground truth CIF...")
    iface_AB, iface_AC = get_interface_residues()

    print("Loading DockQ per seed...")
    dockq_ab, dockq_ac = load_dockq_per_seed()

    cdr_residues = get_cdr_residues()
    print(f"CDR residue counts: { {k: len(v) for k, v in cdr_residues.items()} }")

    print("Computing stats...")
    stats = compute_stats(seed_to_masked, iface_AB, iface_AC, cdr_residues, features)

    # Print table sorted by avg DockQ
    print("\n" + "=" * 90)
    print(f"{'Seed':>5} {'DockQ_AB':>9} {'DockQ_AC':>9} {'Frac_tot':>9} "
          f"{'Frac_AB':>9} {'Frac_AC':>9}  good?")
    print("-" * 90)
    for seed in sorted(SEEDS, key=lambda s: -(dockq_ab.get(s, 0) + dockq_ac.get(s, 0))):
        s = stats[seed]
        flag = " <-- GOOD" if seed in GOOD_SEEDS else ""
        print(f"{seed:>5} {dockq_ab.get(seed, float('nan')):>9.3f} "
              f"{dockq_ac.get(seed, float('nan')):>9.3f} "
              f"{s['frac_total_masked']:>9.3f} "
              f"{s['frac_iface_AB_masked']:>9.3f} "
              f"{s['frac_iface_AC_masked']:>9.3f}{flag}")

    # Summary: good vs bad seeds
    good = [s for s in SEEDS if s in GOOD_SEEDS]
    bad  = [s for s in SEEDS if s not in GOOD_SEEDS]
    print("\nMean stats for good seeds (14,45,47,49):")
    for k in ("frac_total_masked", "frac_iface_AB_masked", "frac_iface_AC_masked"):
        vals = [stats[s][k] for s in good]
        print(f"  {k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    print("Mean stats for bad seeds:")
    for k in ("frac_total_masked", "frac_iface_AB_masked", "frac_iface_AC_masked"):
        vals = [stats[s][k] for s in bad]
        print(f"  {k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    print("\nPlotting scatter...")
    plot_scatter(stats, dockq_ab, dockq_ac)
    plot_cdr_scatter(stats, dockq_ab, dockq_ac)

    print("\nRunning clustering...")
    mat = build_mask_matrix(seed_to_masked, features)
    run_clustering(mat, dockq_ab)


if __name__ == "__main__":
    main()
