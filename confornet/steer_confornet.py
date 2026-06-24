"""ConforNet-style label-free diversity steering for 7txw antibody-antigen docking.

Adapts ConforNets (arXiv:2604.18559) to ESMFold2: instead of steering the raw s_lm, optimize a
tiny channel-wise affine (W: c_z x c_z, b: c_z) on the pair conditioning `lm_z`, with k particles
pushed apart by a CROSS-BLOCK distogram repulsion (our W1). Each "run" is one k-particle diversity
optimization; the script loops CONTINUOUSLY (honest GPU use) and asks: do the k particles land in
*different docking basins*, and does any reach G?

Minimal saving (no quota blow-up): only small CSV rows per particle (basin/wg) + the ~256 KB
affine (W,b) for any real-G hit (wg<0.08). NEVER the 144 MB pairformer z.

Run (esm venv, GPU):
  CUDA_VISIBLE_DEVICES=N python steer_confornet.py --k 2 --tag gpuN --seed-base N000000
"""
import csv
import itertools
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_grad import (  # noqa: E402
    load, compute_z_init, fixed_z0, single_rep_from_hidden, load_hidden,
    lift_to_lm_z, run_trunk, cross_distogram_from_z, w1, BASELINE, N_LOOPS,
)
from basins import load_centroids            # noqa: E402

_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
from paths import CSV_DIR, ESM_PREDICTIONS  # noqa: E402

C_Z = 256
REAL_G = 0.08


class ConforNet(nn.Module):
    """Channel-wise affine on the pair latent: lm_z @ W^T + b  (init = identity / 0)."""
    def __init__(self, dim=C_Z):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, lm_z):
        return torch.matmul(lm_z, self.W.t()) + self.b


def _flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name:
            return cast(sys.argv[sys.argv.index(a) + 1])
    return default


def main():
    k = _flag("--k", 2, int)
    steps = _flag("--steps", 21, int)
    lr = _flag("--lr", 1e-3, float)
    K = _flag("--K", 3, int)
    noise = _flag("--noise", 0.01, float)          # identity-perturbation scale (symmetry break)
    seed_base = _flag("--seed-base", 0, int)
    tag = _flag("--tag", "")
    max_runs = _flag("--max-runs", 0, int)         # 0 = infinite (continuous)

    print(f"loading model... ConforNet k={k} steps={steps} lr={lr} K={K} noise={noise} "
          f"tag={tag} seed_base={seed_base} (max_runs={max_runs or 'inf'})", flush=True)
    model, builder, feat, chain_infos, cond = load()
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)
    s0 = single_rep_from_hidden(model, load_hidden(BASELINE)).detach()
    lm_z0 = lift_to_lm_z(model, s0).float().detach()           # baseline pair conditioning, fixed
    with torch.no_grad():
        Pg, Pb1, Pb2 = load_centroids(model, pair_mask)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    out_csv = CSV_DIR / f"steer_confornet{suffix}.csv"
    fields = ["run", "particle", "basin", "wg", "wb1", "wb2", "repel"]
    write_header = not out_csv.exists()
    fcsv = open(out_csv, "a", newline="")
    wcsv = csv.DictWriter(fcsv, fieldnames=fields)
    if write_header:
        wcsv.writeheader()
    affine_dir = ESM_PREDICTIONS / "7txw_confornet" / (tag or "default")

    def distogram(net, total_steps, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z_out, _ = run_trunk(model, z_init, net(lm_z0), pair_mask, z0, total_steps)
            return cross_distogram_from_z(model, z_out, pair_mask)

    counts = {"G": 0, "B1": 0, "B2": 0}
    realg = 0
    run = 0
    while max_runs == 0 or run < max_runs:
        torch.manual_seed(seed_base + run)
        nets = [ConforNet().cuda() for _ in range(k)]
        for net in nets:                                       # break symmetry: identity + small noise
            with torch.no_grad():
                net.W.add_(torch.randn_like(net.W) * noise)
                net.b.add_(torch.randn_like(net.b) * noise)
        opt = torch.optim.Adam([p for net in nets for p in net.parameters()], lr=lr)
        pairs = list(itertools.combinations(range(k), 2))

        repel = 0.0
        for s in range(steps):
            Ps = [distogram(net, K, grad=True) for net in nets]
            mean_dist = sum(w1(Ps[i], Ps[j]) for i, j in pairs) / len(pairs)
            loss = -mean_dist                                  # maximize pairwise divergence
            opt.zero_grad()
            loss.backward()
            for net in nets:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            repel = mean_dist.item()
            if s > 0 and s % 5 == 0:                           # halve lr every 5 steps
                for g in opt.param_groups:
                    g["lr"] *= 0.5

        # eval each particle at the full horizon -> basin
        run_basins = []
        for i, net in enumerate(nets):
            P17 = distogram(net, N_LOOPS + 1, grad=False)
            wg, wb1, wb2 = (w1(P17, c).item() for c in (Pg, Pb1, Pb2))
            basin = min((("G", wg), ("B1", wb1), ("B2", wb2)), key=lambda kv: kv[1])[0]
            counts[basin] = counts.get(basin, 0) + 1
            run_basins.append((basin, wg))
            wcsv.writerow({"run": run, "particle": i, "basin": basin,
                           "wg": round(wg, 4), "wb1": round(wb1, 4), "wb2": round(wb2, 4),
                           "repel": round(repel, 4)})
            if basin == "G" and wg < REAL_G:                   # save ONLY the tiny affine for a real-G hit
                realg += 1
                ad = affine_dir / f"run{run:05d}_p{i}"
                ad.mkdir(parents=True, exist_ok=True)
                torch.save({"W": net.W.detach().cpu(), "b": net.b.detach().cpu(),
                            "wg": wg, "run": run, "particle": i}, ad / "confornet.pt")
        fcsv.flush()

        sep = len({b for b, _ in run_basins}) > 1              # did the particles split across basins?
        g = "  <== G!" if any(b == "G" and wgv < REAL_G for b, wgv in run_basins) else ""
        print(f"run {run:5d}: basins={[b for b, _ in run_basins]} separated={sep} repel={repel:.3f} "
              f"| totals G={counts['G']} B1={counts['B1']} B2={counts['B2']} realG={realg}{g}",
              flush=True)

        del nets, opt
        if run % 20 == 0:
            torch.cuda.empty_cache()
        run += 1

    if torch.cuda.is_available():
        print(f"peak GPU mem this process: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
              flush=True)


if __name__ == "__main__":
    main()
