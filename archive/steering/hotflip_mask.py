"""One-shot HotFlip mask saliency for 7txw: which input residues, if masked,
move the short-horizon (K=3) cross-block distogram toward the good basin G?

Idea
----
Masking is the *in-distribution* steering lever (the LM mask token 32 is what ESMC
was pretrained on), unlike free perturbation of the continuous `s_lm` which walks
off-manifold (summary.md §8). To rank residues by "does masking this help reach G"
in ONE backward (not 609 forwards), we use a HotFlip score:

    score_i = -( dL/dX_i  .  (E[mask] - E[true_i]) )

where X_i is the ESMC *input embedding* at LM position i, E[mask]=embed.weight[32],
and L = w1(P_K(X), P_G) read at K=3. The dot product is the first-order estimate of
how the only available move (true->mask swap) changes L; we negate so positive = the
swap lowers L (moves toward G). All ESMC weights are frozen; X is the only leaf.

Why attract-to-G (not label-free repulsion): a "repel from baseline" gradient is
degenerate AT the unmasked baseline (you sit on the anchor -> grad 0). Attract-to-G
is non-degenerate and directly tests whether the saliency localizes to the epitope
(antigen C48-C50, the observationally-enriched residues from summary.md §5).

Backward through ESMC is enabled by NOT using compute_lm_hidden_states (it wraps the
LM call in torch.inference_mode + detach). We replicate its BOS/EOS collapse+scatter,
call _esmc without inference_mode, and inject a differentiable embedding leaf via a
forward hook on _esmc.embed. ESMC is differentiable in bf16 (fp8 off).

Run (esm venv, GPU): CUDA_VISIBLE_DEVICES=5 ~/esm/.venv/bin/python \
    analysis/grad_steer/hotflip_mask.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden,
    distogram_from_s_lm, w1, good_reference, AB, C,
)

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "analysis"))
from paths import CSV_DIR, REPO_ROOT  # noqa: E402
from analyze_masks import build_lm_input_ids  # noqa: E402

MASK_TOKEN_ID = 32
K = 3                       # short-horizon read (responsive; summary.md §8)
FIG = REPO_ROOT / "analysis" / "figs" / "hotflip_mask_oneshot.png"
OUT_CSV = CSV_DIR / "hotflip_mask_7txw.csv"


def seq_id_from(lm_input_ids):
    """Chain-aware sequence_id (BOS=0 starts a chain; PAD=1 -> -1). Matches
    compute_lm_hidden_states."""
    sid = (lm_input_ids == 0).cumsum(1) - 1
    return sid.masked_fill(lm_input_ids == 1, -1)


def esmc_hidden(model, lm_input_ids, sequence_id, protein_mask, expand_maps, L, leaf=None):
    """Differentiable replacement for compute_lm_hidden_states (no inference_mode,
    no detach). Returns per-token hidden stack [1, L, n_layers+1, D]. If `leaf` is
    given, a forward hook replaces the embedding output with it (the autograd leaf)."""
    esmc = model._esmc
    handle = esmc.embed.register_forward_hook(lambda m, i, o: leaf) if leaf is not None else None
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = esmc(input_ids=lm_input_ids, sequence_id=sequence_id,
                       output_hidden_states=True)
    finally:
        if handle is not None:
            handle.remove()
    hs = out.hidden_states                       # [n_layers+1, B, max_len, D]
    n_lp1, B, _, D = hs.shape
    result = torch.zeros(B, L, n_lp1, D, device=hs.device, dtype=hs.dtype)
    for b in range(B):
        mb = protein_mask[b]
        em = expand_maps[b][mb]                  # LM positions of this batch's protein tokens
        gathered = hs[:, b, em, :].permute(1, 0, 2)   # [n_prot, n_layers+1, D]
        result[b, mb.nonzero(as_tuple=True)[0]] = gathered
    return result


def distogram_K(model, hidden, z_init, pair_mask, z0):
    s_lm = single_rep_from_hidden(model, hidden)
    P, _ = distogram_from_s_lm(model, s_lm, z_init, pair_mask, z0, total_steps=K)
    return P


def main():
    torch.manual_seed(0)
    print("loading model (ESMC kept on GPU for backward)...", flush=True)
    model, builder, feat, chain_infos, cond = load(keep_esmc=True)
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)
    esmc = model._esmc
    if getattr(esmc, "_use_flash_attn", False):
        print("  WARN: ESMC uses flash-attn; multi-chain sequence_id may raise. "
              "Reload with attn_implementation='sdpa' if it does.", flush=True)

    Pg = good_reference(model, pair_mask, z0, z_init)            # mature-G target (loop16)

    # --- LM input + residue<->LM-position map (same construction as the pipeline) ---
    lm_input_ids, expand_maps = build_lm_input_ids(feat)
    sequence_id = seq_id_from(lm_input_ids)
    mol_type = feat["mol_type"]
    tok_mask = feat["token_attention_mask"]
    protein_mask = (mol_type == 0) & tok_mask.bool()
    L = feat["input_ids"].shape[1]
    print(f"L tokens={L}  lm_input_ids={tuple(lm_input_ids.shape)}  "
          f"protein tokens={int(protein_mask.sum())}", flush=True)

    # --- differentiable forward with the input embeddings as the leaf ---
    with torch.no_grad():
        X0 = esmc.embed(lm_input_ids)                           # true input embeddings [1,Llm,D]
    X = X0.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        hidden = esmc_hidden(model, lm_input_ids, sequence_id, protein_mask,
                             expand_maps, L, leaf=X)
        P_K = distogram_K(model, hidden, z_init, pair_mask, z0)
        loss = w1(P_K, Pg)
    loss.backward()
    G = X.grad                                                  # [1, Llm, D]
    L0 = loss.item()
    print(f"\nloss w1(P_K@{K}, P_G) = {L0:.5f}  ||dL/dX||={G.norm().item():.4e}  "
          f"finite={torch.isfinite(G).all().item()}", flush=True)

    # --- HotFlip score: (mask-move) . gradient, per LM position; +score = toward G ---
    E_mask = esmc.embed.weight[MASK_TOKEN_ID].detach()          # [D]
    M = (E_mask.view(1, 1, -1) - X0).float()                    # [1, Llm, D] true->mask move
    raw = (G.float() * M).sum(-1)                               # [1, Llm]  dL est. of the swap
    score_lm = (-raw)[0]                                        # [Llm]  +ve => masking lowers L

    # --- map LM positions back to residues (7txw: all tokens protein, 1 LM pos each) ---
    em0 = expand_maps[0]                                        # [L] token -> LM pos (>=0)
    score_res = score_lm[em0].detach().float().cpu().numpy()    # [L]
    asym = feat["asym_id"][0].cpu().numpy()
    res_idx = feat["residue_index"][0].cpu().numpy()
    chains = np.array([chr(ord("A") + int(a)) for a in asym])

    # ------------------------------------------------------------ diagnostics
    ab_abs = np.abs(score_res[:438]).mean()
    ag_abs = np.abs(score_res[438:]).mean()
    print(f"\nmean |score|: antibody(0-437)={ab_abs:.3e}  antigen C(438-608)={ag_abs:.3e}  "
          f"(expect antigen >> antibody per §4)", flush=True)

    order = np.argsort(-score_res)                              # most-helpful first
    print("\ntop-12 residues by HotFlip score (toward-G):", flush=True)
    for t in order[:12]:
        print(f"  tok {t:3d}  {chains[t]}{int(res_idx[t])+1:<4d} score={score_res[t]:+.3e}",
              flush=True)

    # rank of the §5 candidates C48-C50 (asym 2, residue_index 47/48/49)
    cand_tok = [t for t in range(L) if asym[t] == 2 and res_idx[t] in (47, 48, 49)]
    rank_of = {int(t): int(np.where(order == t)[0][0]) for t in cand_tok}
    print(f"\nC48-C50 tokens {cand_tok}  ranks (0=best of {L}): "
          f"{[rank_of[t] for t in cand_tok]}", flush=True)

    # --------------------------------------------- VALIDATION (1): along-gradient
    def loss_at_leaf(leaf):
        with torch.no_grad():
            hid = esmc_hidden(model, lm_input_ids, sequence_id, protein_mask,
                              expand_maps, L, leaf=leaf)
            return w1(distogram_K(model, hid, z_init, pair_mask, z0), Pg).item()

    jit = abs(loss_at_leaf(X0) - loss_at_leaf(X0))
    gn = G.norm().item()
    ghat = (G / (gn + 1e-12)).detach()
    print(f"\nalong-gradient check (leaf = X0 - eps*ghat)   noise floor={jit:.2e}", flush=True)
    print(f"  {'eps':>6} {'pred dL':>12} {'actual dL':>12} {'ratio':>7}", flush=True)
    for eps in (0.5, 2.0, 5.0):
        actual = loss_at_leaf(X0 - eps * ghat) - L0
        pred = -eps * gn
        print(f"  {eps:6.2f} {pred:12.4e} {actual:12.4e} {actual/pred:7.2f}", flush=True)

    # --------------------------------------------- VALIDATION (2): real top-k mask
    def loss_with_masked(token_positions):
        ids = lm_input_ids.clone()
        lp = [int(em0[t]) for t in token_positions]
        ids[0, lp] = MASK_TOKEN_ID
        with torch.no_grad():
            hid = esmc_hidden(model, ids, sequence_id, protein_mask, expand_maps, L)
            return w1(distogram_K(model, hid, z_init, pair_mask, z0), Pg).item()

    k = 10
    topk = [int(t) for t in order[:k]]
    rng = np.random.default_rng(0)
    antigen_tokens = np.where(asym == 2)[0]
    randk = [int(t) for t in rng.choice(antigen_tokens, size=k, replace=False)]
    dL_top = loss_with_masked(topk) - L0
    dL_rand = loss_with_masked(randk) - L0
    print(f"\nreal-mask validation (actually set token 32, K={K}):", flush=True)
    print(f"  baseline L0={L0:.4f}", flush=True)
    print(f"  mask top-{k} (HotFlip)   dL={dL_top:+.4f}  (want < 0, toward G)", flush=True)
    print(f"  mask random-{k} antigen  dL={dL_rand:+.4f}", flush=True)

    # ------------------------------------------------------------------- CSV + plot
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT_CSV, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["token", "chain", "residue_1based", "hotflip_score_towardG"])
        for t in range(L):
            wr.writerow([t, chains[t], int(res_idx[t]) + 1, f"{score_res[t]:.6e}"])
    print(f"\nwrote {OUT_CSV}", flush=True)

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(L)
    colors = np.where(x < 438, "tab:gray", "tab:blue")
    ax.bar(x, score_res, color=colors, width=1.0)
    ax.axvline(438, color="black", ls="--", lw=1, label="antibody | antigen (C)")
    for t in cand_tok:
        ax.axvline(t, color="red", lw=1.2, alpha=0.8)
    ax.scatter([], [], color="red", marker="|", label="C48-C50 (epitope-adjacent, §5)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("residue token (0-437 antibody A/B, 438-608 antigen C)")
    ax.set_ylabel("HotFlip score  (+ = masking moves toward G)")
    ax.set_title(f"7txw one-shot HotFlip mask saliency  (K={K}, attract-to-G)  "
                 f"| C48-C50 ranks {[rank_of[t] for t in cand_tok]} of {L}")
    ax.legend(fontsize=8, loc="upper left")
    ax.margins(x=0.005)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}", flush=True)
    print("\nhotflip_mask OK.", flush=True)


if __name__ == "__main__":
    main()
