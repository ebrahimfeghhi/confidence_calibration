"""Phase 2 of the steer->launch->decode pipeline: deferred batched launch + decode.

Loads leaves_{tag}.npy (optimized s_lm's saved by repel_proven.py), batches the
forward-only 0->17 launch (final_trunk_z_batched, chunked by --launch-batch), then
decodes each launched z_trunk -> structure -> per-(run,sample) iptm + ab-ag DockQ, and
logs decode-phase wall-clock + peak VRAM. The diffusion decode is per-structure
(VRAM-bound), so only the launch is batched; the decode sub-batches at 1.

Run (esm venv, GPU): CUDA_VISIBLE_DEVICES=N python decode_batch.py --tag 7txw_k3 --target 7txw
"""
import sys
import csv
import time
import json
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from forward_grad import (load, compute_z_init, fixed_z0, lift_to_lm_z,          # noqa: E402
                          final_trunk_z, final_trunk_z_batched, decode_from_ztrunk,
                          decode_from_ztrunk_batched, N_LOOPS)
import forward_grad as fg  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from eval_ost_foldbench import postprocess_cif, OST_BIN, OST_ENV  # noqa: E402
from paths import GROUND_TRUTHS, TARGETS_CSV  # noqa: E402

OUT = Path(__file__).resolve().parent


def flag(name, default, cast=str):
    for a in sys.argv[1:]:
        if a == name and cast == bool:
            return True
        if a.startswith(name + "="):
            return cast(a.split("=", 1)[1])
        if a == name and cast != bool:
            return cast(sys.argv[sys.argv.index(a) + 1])
    return default


def main():
    tag = flag("--tag", "7txw_k3", str)
    target = flag("--target", "7txw", str)
    n_samples = flag("--n-samples", 5, int)
    n_steps = flag("--n-steps", 200, int)
    launch_batch = flag("--launch-batch", 4, int)
    skip_dockq = flag("--skip-dockq", False, bool)
    fused = flag("--fused", False, bool)                 # fused Triton trimul kernel (method-preserving)
    chunk_size = flag("--chunk-size", 32, int)           # trunk chunk_size (<=0 -> None)
    batched_decode = flag("--batched-decode", False, bool)  # decode each launch chunk's structures in one diffusion call
    print(f"decode_batch: tag={tag} target={target} n_samples={n_samples} n_steps={n_steps} "
          f"launch_batch={launch_batch} batched_decode={batched_decode} fused={fused} "
          f"chunk={chunk_size} dockq={not skip_dockq}", flush=True)

    fg.configure_target(target)
    NATIVE = GROUND_TRUTHS / f"{target}-assembly1.cif"

    # --- ab-ag DockQ helper (antigen from FoldBench metadata; excludes heavy-light) ---
    antigen, want = "C", set()
    try:
        with open(TARGETS_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("pdb_id", "").split("-")[0] == target:
                    want.update(c for c in (row.get("interface_chain_id_1"),
                                            row.get("interface_chain_id_2")) if c)
                    if row.get("antigen_chain"):
                        antigen = row["antigen_chain"].strip()
    except Exception:
        pass
    FORCE_MAP = [f"{c}:{c}" for c in sorted(want)] or ["A:A", "B:B", "C:C"]

    def _ost(patched, outj, chain_map=None):
        cmd = [OST_BIN, "compare-structures", "-m", str(patched), "-r", str(NATIVE),
               "-o", str(outj), "--fault-tolerant", "--min-pep-length", "4",
               "--min-nuc-length", "4", "--lddt", "--dockq"]
        if chain_map:
            cmd += ["-c", *chain_map]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=OST_ENV)
        except Exception:
            return False
        return r.returncode == 0 and outj.exists() and outj.stat().st_size > 0

    def _abag(outj):
        d = json.load(open(outj))
        ifaces = list(zip(d.get("dockq_interfaces", []), d.get("dockq", [])))
        abag = [dq for iface, dq in ifaces if antigen in {c.split("-")[0] for c in iface[:2]}]
        pool = abag if abag else [dq for _, dq in ifaces]   # 2-chain VHH: single interface
        return max(pool) if pool else None

    def dockq_cif(cif):
        try:
            with tempfile.TemporaryDirectory() as td:
                patched, outj = Path(td) / "m.cif", Path(td) / "o.json"
                if not postprocess_cif(Path(cif), patched):
                    return float("nan")
                dq = _abag(outj) if _ost(patched, outj) else None
                if dq is None:                              # unusable -> force identity map & re-parse
                    dq = _abag(outj) if _ost(patched, outj, FORCE_MAP) else None
                return dq if dq is not None else float("nan")
        except Exception:
            return float("nan")

    model, builder, feat, chain_infos, cond = load(chunk_size=chunk_size,
                                                    kernel_backend="fused" if fused else None)
    pair_mask = cond["pair_mask"]
    z_init = compute_z_init(model, cond)
    z0 = fixed_z0(model, z_init, seed=0)

    leaves = np.load(OUT / f"leaves_{tag}.npy")             # [N,1,L,256]
    N = leaves.shape[0]
    print(f"loaded {N} leaves {tuple(leaves.shape)}", flush=True)

    ddir = OUT / f"decoded_{tag}"
    ddir.mkdir(exist_ok=True)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for c0 in range(0, N, launch_batch):
        chunk = list(range(c0, min(c0 + launch_batch, N)))
        lm_z_B = torch.cat([lift_to_lm_z(model, torch.from_numpy(leaves[i].astype(np.float32)).cuda())
                            for i in chunk], dim=0)          # [B,L,L,256]
        try:
            z_B = final_trunk_z_batched(model, z_init, lm_z_B, pair_mask, z0, N_LOOPS + 1)
            z_list = [z_B[bi:bi + 1] for bi in range(len(chunk))]
        except Exception as e:                  # batched launch unvalidated -> safe proven per-leaf fallback
            print(f"  batched launch failed ({type(e).__name__}: {e}); per-leaf fallback", flush=True)
            z_B = None
            z_list = [final_trunk_z(model, z_init, lm_z_B[bi:bi + 1], pair_mask, z0, N_LOOPS + 1)
                      for bi in range(len(chunk))]
        samples_per = None
        if batched_decode and z_B is not None:  # decode the whole chunk in ONE diffusion call (batch B*n_samples)
            try:
                samples_per = decode_from_ztrunk_batched(model, builder, feat, chain_infos, cond,
                                                         z_B, n_samples, n_steps, target, seed=c0 + 1)
            except Exception as e:
                print(f"  batched decode failed ({type(e).__name__}: {e}); per-leaf fallback", flush=True)
                samples_per = None
        if samples_per is None:                 # per-leaf decode (default / fallback), seed=run
            samples_per = [decode_from_ztrunk(model, builder, feat, chain_infos, cond,
                                              z_list[bi], n_samples, n_steps, target, seed=i + 1)
                           for bi, i in enumerate(chunk)]
        for bi, i in enumerate(chunk):
            run = i + 1
            samples = samples_per[bi]
            for k, s in enumerate(samples):
                cif = ddir / f"run{run}_s{k}.cif"
                cif.write_text(s.complex.to_mmcif())
                ip = float(s.iptm) if s.iptm is not None else (float(s.ptm) if s.ptm is not None else 0.0)
                dq = float("nan") if skip_dockq else dockq_cif(cif)
                rows.append(dict(run=run, sample=k, iptm=round(ip, 4), dockq=round(dq, 4)))
            best = max((r["dockq"] for r in rows if r["run"] == run and r["dockq"] == r["dockq"]),
                       default=float("nan"))
            print(f"  run {run:2d}: iptm={[r['iptm'] for r in rows if r['run'] == run]} "
                  f"best_dockq={best:.3f}", flush=True)
            with open(OUT / f"dockq_{tag}.csv", "w", newline="") as df:   # incremental flush
                w = csv.DictWriter(df, fieldnames=["run", "sample", "iptm", "dockq"])
                w.writeheader(); w.writerows(rows)
        del lm_z_B, z_list
        torch.cuda.empty_cache()
    decode_secs = round(time.time() - t0, 1)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    with open(OUT / f"decode_cost_{tag}.csv", "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=["tag", "n_leaves", "n_samples", "n_steps",
                                           "launch_batch", "decode_secs", "peak_gb"])
        w.writeheader()
        w.writerow(dict(tag=tag, n_leaves=N, n_samples=n_samples, n_steps=n_steps,
                        launch_batch=launch_batch, decode_secs=decode_secs, peak_gb=peak_gb))
    print(f"[{tag}] decode done {decode_secs}s, peak {peak_gb} GB, {len(rows)} (run,sample) rows "
          f"-> dockq_{tag}.csv", flush=True)


if __name__ == "__main__":
    main()
