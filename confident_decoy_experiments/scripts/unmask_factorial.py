#!/usr/bin/env python
"""UNMASKED-TEMPLATE FACTORIAL on 8che: MSA {none, subsampled, full} x MC dropout {on, off}.

WHAT IS BEING VARIED. All six cells use the SAME donor template (137_3 = seed 137 sample 3, the
dockqdonor pick for 8che_BE, own DockQ 0.529) and the SAME 10 seeds as allpose8che10's 137_3 arm, so
every cell is seed- and template-matched to the existing MASKED baseline (best 0.957, 6/10 seeds
near-native). All six run from /home/ebrahim/Protenix_unmask, where multichain_mask is forced to ones
so template CROSS-CHAIN geometry reaches the pair representation. /home/ebrahim/Protenix is untouched.

  MSA axis
    none        both msa paths point at a 1-sequence a3m (the query). Deleting the paths does NOT
                work -- msa_search.py:44 treats absent paths as "missing" and launches a fresh MSA
                search, which is the opposite of the intended condition.
    subsampled  stock behaviour: rows drawn UNIFORMLY from [1, n] every trunk pass (utils.py:306)
    full        --data.msa.min_size.test 16384 forces sample_size == n, i.e. no subsampling
  Dropout axis
    on          stock: mc_dropout_apply_rate 0.4 (configs_base.py:109). F.dropout at protenix.py:242
                has training=True by DEFAULT and it is not passed, so model.eval() does NOT stop it.
    off         --mc_dropout_apply_rate 0

EVERY CELL PROVES ITS OWN CONDITION. A masked/unmasked, dropout/no-dropout, or MSA-depth difference is
invisible in the output -- three earlier arms in this project were scored and believed while silently
untemplated. So each log is parsed for:
    UNMASKED TEMPLATES ACTIVE ... mean 1.0000     (mask really off)
    MC_DROPOUT rate=<r> ... cumulative=<f>/<n>    (dropout really on/off, per seed)
    N_msa <k>                                     (MSA depth really none/subsampled/full)
    Found 1 templates                             (the template really loaded)
A cell whose log does not match its declared condition is reported FAILED, never scored.

Usage:  /home/ebrahim/Protenix/.venv/bin/python scripts/unmask_factorial.py [--gpu 7] [--cells A,B,..]
Writes: artifacts/sweep_out/unmaskfac_<cell>/, artifacts/unmaskfac_logs/<cell>.log,
        artifacts/unmask_factorial.json
"""
import argparse, json, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import EXP

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="7")
ap.add_argument("--cells", default="A,B,C,D,E,F")
A = ap.parse_args()

ART = f"{EXP}/artifacts"
TREE = os.environ.get("PROTENIX_UNMASK_TREE",
        "/home/ebrahim/tts_abag/confident_decoy_experiments/vendor/Protenix_unmask")
PY = "/home/ebrahim/Protenix/.venv/bin/python"
SEEDS = "137,108,130,250,332,832,1036,1040,1074,1076"
JSON_MSA = f"{ART}/sweep_tmpl/allpose8che/8che_allpose_137_3.json"
JSON_NOMSA = f"{ART}/sweep_tmpl/unmask8che_nomsa/8che_unmask_nomsa_137_3.json"
LOGD = f"{ART}/unmaskfac_logs"
os.makedirs(LOGD, exist_ok=True)

# cell: (msa condition, dropout condition, input json, extra flags)
CELLS = {
    "A": ("none",       "on",  JSON_NOMSA, []),
    "B": ("none",       "off", JSON_NOMSA, ["--mc_dropout_apply_rate", "0"]),
    "C": ("subsampled", "on",  JSON_MSA,   []),
    "D": ("subsampled", "off", JSON_MSA,   ["--mc_dropout_apply_rate", "0"]),
    "E": ("full",       "on",  JSON_MSA,   ["--data.msa.min_size.test", "16384"]),
    "F": ("full",       "off", JSON_MSA,   ["--data.msa.min_size.test", "16384",
                                            "--mc_dropout_apply_rate", "0"]),
}


def log(m):
    print(f"[{time.strftime('%H:%M')}] {m}", flush=True)


def verify(cell, logf):
    """Parse the log and confirm the cell actually ran its declared condition."""
    msa_cond, drop_cond, _, _ = CELLS[cell]
    txt = open(logf, errors="ignore").read()
    v = {"unmasked": None, "mask_mean": None, "n_msa": None,
         "dropout_fired": None, "dropout_seeds": None, "rate": None, "templates": 0}

    m = re.search(r"UNMASKED TEMPLATES ACTIVE.*?([\d.]+) mean", txt)
    v["unmasked"] = bool(m); v["mask_mean"] = float(m.group(1)) if m else None
    v["templates"] = len(re.findall(r"Found 1 templates", txt))
    nm = re.findall(r"N_msa (\d+)", txt)
    v["n_msa"] = sorted({int(x) for x in nm}) if nm else None
    mc = re.findall(r"MC_DROPOUT rate=([\d.]+) this_seed=(ON|off) cumulative=(\d+)/(\d+)", txt)
    if mc:
        v["rate"] = float(mc[-1][0]); v["dropout_fired"] = int(mc[-1][2])
        v["dropout_seeds"] = int(mc[-1][3])

    problems = []
    if not v["unmasked"] or (v["mask_mean"] or 0) < 1.0:
        problems.append("mask NOT forced to ones")
    if v["templates"] == 0:
        problems.append("template never loaded ('Found 0 templates')")
    if v["n_msa"] is None:
        problems.append("no N_msa line")
    else:
        mx = max(v["n_msa"])
        if msa_cond == "none" and mx > 2:
            problems.append(f"MSA should be single-sequence, saw N_msa {v['n_msa']}")
        if msa_cond != "none" and mx < 2:
            problems.append(f"MSA should be deep, saw N_msa {v['n_msa']}")
    if v["dropout_seeds"] is None:
        problems.append("no MC_DROPOUT line")
    else:
        if drop_cond == "off" and (v["rate"] != 0.0 or v["dropout_fired"] != 0):
            problems.append(f"dropout should be OFF, rate={v['rate']} fired={v['dropout_fired']}")
        if drop_cond == "on" and v["rate"] == 0.0:
            problems.append("dropout should be ON but rate=0")
    v["problems"] = problems
    v["ok"] = not problems
    return v


results = {}
for cell in A.cells.split(","):
    cell = cell.strip()
    msa_cond, drop_cond, jsn, extra = CELLS[cell]
    out = f"{ART}/sweep_out/unmaskfac_{cell}"
    logf = f"{LOGD}/{cell}.log"
    os.makedirs(out, exist_ok=True)
    log(f"cell {cell}: MSA={msa_cond} dropout={drop_cond}")
    cmd = [PY, "runner/inference.py", "--model_name", "protenix-v2",
           "--load_checkpoint_dir", "/home/ebrahim/checkpoint",
           "--input_json_path", jsn, "--dump_dir", out, "--seeds", SEEDS,
           "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
           "--use_template", "true"] + extra
    env = dict(os.environ, PYTHONPATH=TREE, CUDA_VISIBLE_DEVICES=A.gpu,
               LAYERNORM_TYPE="openfold", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    t0 = time.time()
    with open(logf, "w") as fh:
        subprocess.run(cmd, cwd=TREE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    v = verify(cell, logf)
    v.update({"cell": cell, "msa": msa_cond, "dropout": drop_cond,
              "minutes": round((time.time() - t0) / 60, 1), "out": out})
    results[cell] = v
    log(f"cell {cell} done in {v['minutes']} min | "
        + ("VERIFIED" if v["ok"] else "FAILED: " + "; ".join(v["problems"]))
        + f" | N_msa={v['n_msa']} dropout {v['dropout_fired']}/{v['dropout_seeds']} @rate {v['rate']}")
    json.dump(results, open(f"{ART}/unmask_factorial.json", "w"), indent=2)

log("ALL CELLS DONE")
for c, v in results.items():
    log(f"  {c}: MSA={v['msa']:>10} dropout={v['dropout']:>3} -> "
        + ("VERIFIED" if v["ok"] else "FAILED " + "; ".join(v["problems"])))
print(f"\nWROTE {ART}/unmask_factorial.json")
