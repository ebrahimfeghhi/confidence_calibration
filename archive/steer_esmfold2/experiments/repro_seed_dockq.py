"""repro_seed_dockq.py — reproducibility of ONE seed: fold it N times (dropout ON, no steering) and
report ipTM + DockQ per trial as each finishes.

Dropout/mask/z0 are all deterministic given the seed, so re-folding the SAME seed varies ONLY by CUDA
non-determinism. For a knife-edge seed (barely tips into the good basin) that noise can flip the outcome
in/out of the good dock — this shows how often it lands good. Each trial folds with the real --seed but is
saved to seed_<trial> so the OST scorer treats each as its own structure. DockQ scored inline via OST after
each fold (skips already-done dirs), so results stream out one trial at a time.

  CUDA_VISIBLE_DEVICES=3 python steer/experiments/repro_seed_dockq.py --target 8ts0 --seed 617 --trials 10
"""
import sys, json, argparse, subprocess, csv
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run" / "esm"))
import forward_grad as fg  # noqa: E402

ROOT=Path("/home/ebrahim/tts_abag"); OUT_ROOT=Path.home()/"esm/foldbench_predictions"
PYEXE="/home/ebrahim/esm/.venv/bin/python"
SCRATCH=Path("/tmp/claude-1004/-home-ebrahim-tts-abag/ea5db07e-452e-45f8-a6d6-98121d6d1264/scratchpad")

def best_dockq(csv_path, target, trial):
    if not csv_path.exists(): return None
    best=None
    for r in csv.DictReader(open(csv_path)):
        if r['assembly_id'][:4]!=target or str(r['seed'])!=str(trial): continue
        try: dq=float(r['dockq'])
        except: continue
        if best is None or dq>best: best=dq
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target",default="8ts0"); ap.add_argument("--seed",type=int,default=617)
    ap.add_argument("--trials",type=int,default=10); ap.add_argument("--run-name",default=None)
    ap.add_argument("--mask",type=float,default=0.1); ap.add_argument("--dropout",type=float,default=0.3)
    ap.add_argument("--samples",type=int,default=5)
    a=ap.parse_args()
    run_name=a.run_name or f"repro_{a.target}_{a.seed}"
    fg.configure_target(a.target)
    model,builder,feat,chain_infos,cond=fg.load(keep_esmc=True, kernel_backend="fused")
    spi=fg.load_complex(str(fg.JSON_DIR), fg._TARGET)
    out_dir=OUT_ROOT/run_name/a.target
    csv_path=ROOT/"results/csv_files"/f"dockq_{run_name}.csv"
    ost_log=SCRATCH/f"ost_{run_name}.log"
    print(f"=== reproducibility: {a.trials} folds of {a.target} seed {a.seed} "
          f"(mask {a.mask}, dropout {a.dropout}, {a.samples} samples each) ===",flush=True)
    print(f"{'trial':>5} {'best_ipTM':>10} {'best_DockQ':>11}",flush=True)
    for t in range(a.trials):
        pred_dir=out_dir/f"seed_{t}"/"predictions"
        if pred_dir.exists() and any(pred_dir.glob(f"{a.target}_sample_*.cif")):
            iptm=max((json.load(open(j)).get("iptm") or 0.0)
                     for j in pred_dir.glob(f"{a.target}_summary_confidence_sample_*.json"))
        else:
            result=builder.fold(model,spi,seed=a.seed,num_loops=16,num_sampling_steps=16,
                                num_diffusion_samples=a.samples,lm_mask_pct=a.mask,lm_dropout=a.dropout)
            samples=result if isinstance(result,list) else [result]
            pred_dir.mkdir(parents=True,exist_ok=True)
            for k,s in enumerate(samples):
                (pred_dir/f"{a.target}_sample_{k}.cif").write_text(s.complex.to_mmcif())
                conf={"ranking_score":s.iptm,"iptm":s.iptm,"ptm":s.ptm,
                      "chain_pair_iptm":s.pair_chains_iptm.tolist() if s.pair_chains_iptm is not None else None}
                json.dump(conf,open(pred_dir/f"{a.target}_summary_confidence_sample_{k}.json","w"))
            iptm=max((s.iptm or 0.0) for s in samples); torch.cuda.empty_cache()
        # inline OST DockQ (scores this new seed dir; skips done ones)
        with open(ost_log,"a") as lg:
            subprocess.run([PYEXE,"eval/eval_ost_foldbench.py","--pred_dir",str(out_dir.parent),
                            "--out_csv",str(csv_path)],cwd=str(ROOT),stdout=lg,stderr=lg)
        dq=best_dockq(csv_path,a.target,t)
        print(f"{t:>5} {iptm:>10.3f} {('%.3f'%dq) if dq is not None else '   n/a':>11}",flush=True)
    print("ALL DONE",flush=True)

if __name__=="__main__":
    main()
