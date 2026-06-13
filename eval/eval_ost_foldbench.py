"""
Evaluate antibody-antigen DockQ using FoldBench's exact methodology.
Saves DockQ for ALL diffusion samples per seed (not just the best-ranked one),
so any selection strategy can be tested in post-processing without re-running OST.

Key design:
  - OST is run ONCE per (structure, seed, sample) even for multi-interface structures,
    then DockQ is extracted for each interface from the single OST output.
  - chain_mapping (ref→model chain dict) is saved for reference.

Output: one row per (assembly_id, chain1, chain2, seed, sample)
Columns: assembly_id, chain1, chain2, seed, sample,
         dockq,                 # interface DockQ (primary metric)
         pair_iptm,             # this interface's chain-pair ipTM
         global_iptm,           # whole-complex ipTM
         ptm,                   # whole-complex pTM
         pair_ranking_score,    # 0.8*pair_iptm   + 0.2*ptm - 100*clash
         global_ranking_score,  # 0.8*global_iptm + 0.2*ptm - 100*clash
         chain_mapping          # OST ref->model chain map (debug)

Usage:
  python eval_ost_foldbench.py --pred_dir foldbench_abag/predictions_template
"""
import argparse
import csv
import json
import os
import subprocess
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# --- repo-root bootstrap so `from paths import ...` works from any cwd ---
import sys
_p = Path(__file__).resolve()
while not (_p / "paths.py").exists():
    _p = _p.parent
sys.path.insert(0, str(_p))
import paths

PROTENIX_PYTHON = str(paths.PROTENIX_PYTHON)
OST_BIN = str(paths.OST_BIN)
OST_ENV = {**os.environ, "LD_LIBRARY_PATH": str(paths.OST_LIB)}

GROUND_TRUTH = paths.GROUND_TRUTHS
TARGETS_CSV  = paths.TARGETS_CSV

MAX_WORKERS = 100
FIELDNAMES  = ["assembly_id", "chain1", "chain2", "seed", "sample",
               "dockq", "pair_iptm", "global_iptm", "ptm",
               "pair_ranking_score", "global_ranking_score", "chain_mapping"]

POSTPROCESS_SCRIPT = """
import sys
import biotite.structure.io.pdbx as pdbx
src, dst = sys.argv[1], sys.argv[2]
cif_file = pdbx.CIFFile.read(src)
block = cif_file.block
atom_site = block.get('atom_site')
n = len(atom_site['group_PDB'])
atom_site['occupancy']      = pdbx.CIFColumn(pdbx.CIFData(['1'] * n))
atom_site['B_iso_or_equiv'] = pdbx.CIFColumn(pdbx.CIFData(['0'] * n))
dict_entity_id = {}
for eid, grp in zip(atom_site['label_entity_id'].as_array().tolist(),
                    atom_site['group_PDB'].as_array().tolist()):
    if eid not in dict_entity_id:
        dict_entity_id[eid] = ('polymer' if grp == 'ATOM' else 'non-polymer', grp)
    elif dict_entity_id[eid][1] != grp:
        dict_entity_id[eid] = ('polymer', grp)
block['entity'] = pdbx.CIFCategory({
    'id':   list(dict_entity_id.keys()),
    'type': [dict_entity_id[k][0] for k in dict_entity_id.keys()],
})
cif_file.write(dst)
"""


try:
    import biotite.structure.io.pdbx as _pdbx
    _HAVE_BIOTITE = True
except Exception:
    _HAVE_BIOTITE = False


def _postprocess_cif_inproc(src: Path, dst: Path) -> bool:
    """Same patch as POSTPROCESS_SCRIPT, but in-process (no subprocess)."""
    cif_file = _pdbx.CIFFile.read(str(src))
    block = cif_file.block
    atom_site = block.get("atom_site")
    n = len(atom_site["group_PDB"])
    atom_site["occupancy"]      = _pdbx.CIFColumn(_pdbx.CIFData(["1"] * n))
    atom_site["B_iso_or_equiv"] = _pdbx.CIFColumn(_pdbx.CIFData(["0"] * n))
    dict_entity_id = {}
    for eid, grp in zip(atom_site["label_entity_id"].as_array().tolist(),
                        atom_site["group_PDB"].as_array().tolist()):
        if eid not in dict_entity_id:
            dict_entity_id[eid] = ("polymer" if grp == "ATOM" else "non-polymer", grp)
        elif dict_entity_id[eid][1] != grp:
            dict_entity_id[eid] = ("polymer", grp)
    block["entity"] = _pdbx.CIFCategory({
        "id":   list(dict_entity_id.keys()),
        "type": [dict_entity_id[k][0] for k in dict_entity_id.keys()],
    })
    cif_file.write(str(dst))
    return True


def postprocess_cif(src: Path, dst: Path) -> bool:
    """Patch occupancy/B-factor/entity so OST can read a predicted CIF.

    Done in-process when biotite is importable (faster, and the eval no longer
    needs the Protenix venv just for this); otherwise falls back to running the
    same code via a PROTENIX_PYTHON subprocess.
    """
    if _HAVE_BIOTITE:
        try:
            return _postprocess_cif_inproc(src, dst)
        except Exception:
            pass  # any in-process failure -> try the subprocess path below
    r = subprocess.run(
        [PROTENIX_PYTHON, "-c", POSTPROCESS_SCRIPT, str(src), str(dst)],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0


def run_ost(model_cif: Path, native_cif: Path, out_json: Path) -> bool:
    cmd = [
        OST_BIN, "compare-structures",
        "-m", str(model_cif), "-r", str(native_cif), "-o", str(out_json),
        "--fault-tolerant",
        "--min-pep-length", "4", "--min-nuc-length", "4",
        "--lddt", "--rigid-scores", "--tm-score", "--dockq",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=OST_ENV)
    return r.returncode == 0


def chain_to_asym_id(chain_id: str) -> int:
    """Convert a chain label (e.g. 'A', 'AA') to a 0-based index into chain_pair_iptm."""
    base = chain_id.split("-")[0]
    idx = 0
    for ch in base:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def compute_confidence_scores(conf: dict, chain_mapping: dict, chain1: str, chain2: str) -> dict:
    """Return pair_iptm, pair_ranking_score, global_iptm, ptm, global_ranking_score.

    Uses OST chain_mapping (ref->model) so that pair_iptm correctly indexes
    chain_pair_iptm even when model chains are not in the same order as reference chains.
    """
    nan = float("nan")
    ptm       = conf.get("ptm", 0)
    has_clash = conf.get("has_clash", 0)

    matrix = conf.get("chain_pair_iptm", [])
    if matrix and chain_mapping:
        # chain_mapping keys are the exact reference chain names (including any
        # copy suffix like "A-2"), which match the targets CSV — look them up
        # directly. Do NOT strip the suffix: for a same-letter homo-oligomer that
        # would collapse distinct keys ("A" and "A-2") onto a single model chain
        # and read the wrong (diagonal) chain_pair_iptm entry.
        m1 = chain_mapping.get(chain1)
        m2 = chain_mapping.get(chain2)
        if m1 is not None and m2 is not None:
            try:
                i, j = chain_to_asym_id(m1), chain_to_asym_id(m2)
                pair_iptm = (matrix[i][j] + matrix[j][i]) / 2
                pair_ranking_score = 0.8 * pair_iptm + 0.2 * ptm - 100 * has_clash
            except (IndexError, TypeError):
                pair_iptm = pair_ranking_score = nan
        else:
            pair_iptm = pair_ranking_score = nan
    else:
        pair_iptm = pair_ranking_score = nan

    global_iptm = conf.get("iptm", nan)
    global_ranking_score = (
        0.8 * global_iptm + 0.2 * ptm - 100 * has_clash
        if global_iptm == global_iptm else nan
    )

    return {
        "pair_iptm": pair_iptm,
        "pair_ranking_score": pair_ranking_score,
        "global_iptm": global_iptm,
        "ptm": ptm,
        "global_ranking_score": global_ranking_score,
    }


def extract_dockq(ost_data: dict, chain1: str, chain2: str) -> float:
    """Extract DockQ for a reference interface by exact, order-independent chain match.

    Match on the full reference labels (e.g. "A", "B-2"): OST's dockq_interfaces ref
    slots, the targets CSV, and chain_mapping all use the same exact names, so there is
    no need to strip the copy suffix — and stripping would conflate the separate
    interfaces of a same-letter homo-oligomer (A, A-2, A-3 ...) into one.
    """
    for i, iface in enumerate(ost_data.get("dockq_interfaces", [])):
        c0, c1 = iface[0], iface[1]
        if (c0 == chain1 and c1 == chain2) or (c0 == chain2 and c1 == chain1):
            return ost_data["dockq"][i]
    return float("nan")


def evaluate_group(task):
    """Run OST once for a (pdb_base, seed, sample) and return one row per interface.

    iface_list: [(assembly_id, chain1, chain2), ...]
    All interfaces share the same model CIF and native CIF, so OST runs once for all.
    """
    pdb_base, seed, sample, model_cif, native_cif, conf, iface_list = task
    ost_data = None
    proc_cif = ost_json = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".cif",  delete=False) as tf: proc_cif = Path(tf.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf: ost_json = Path(tf.name)
        if postprocess_cif(model_cif, proc_cif):
            if run_ost(proc_cif, native_cif, ost_json):
                ost_data = json.loads(ost_json.read_text())
    except Exception as e:
        tqdm.write(f"ERROR {pdb_base} seed={seed} sample={sample}: {e}")
    finally:
        if proc_cif and proc_cif.exists(): proc_cif.unlink()
        if ost_json and ost_json.exists(): ost_json.unlink()

    chain_mapping     = ost_data.get("chain_mapping", {}) if ost_data else {}
    chain_mapping_str = json.dumps(chain_mapping) if ost_data else ""

    rows = []
    for assembly_id, chain1, chain2 in iface_list:
        dockq_score = extract_dockq(ost_data, chain1, chain2) if ost_data else float("nan")
        scores      = compute_confidence_scores(conf, chain_mapping, chain1, chain2)
        rows.append({
            "assembly_id": assembly_id, "chain1": chain1, "chain2": chain2,
            "seed": seed, "sample": sample, **scores,
            "dockq": dockq_score, "chain_mapping": chain_mapping_str,
        })
    return rows


def load_interfaces():
    """Returns pdb_base -> [(assembly_id, chain1, chain2, native_cif)]."""
    pdb_to_ifaces = defaultdict(list)
    with open(TARGETS_CSV) as f:
        for row in csv.DictReader(f):
            pdb_id     = row["pdb_id"]
            chain1     = row["interface_chain_id_1"]
            chain2     = row["interface_chain_id_2"]
            pdb_base   = pdb_id.split("-")[0]
            native_cif = GROUND_TRUTH / f"{pdb_id}.cif"
            if not native_cif.exists():
                print(f"  WARNING: no ground truth for {pdb_id}, skipping")
                continue
            pdb_to_ifaces[pdb_base].append((pdb_id, chain1, chain2, native_cif))
    return pdb_to_ifaces


def load_done(out_csv):
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                try:
                    done.add((row["assembly_id"], row["chain1"], row["chain2"],
                              row["seed"], int(row["sample"])))
                except (ValueError, KeyError):
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=str(paths.PROTENIX_PREDICTIONS))
    parser.add_argument("--out_csv",  default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_csv  = Path(args.out_csv) if args.out_csv else \
               pred_dir.parent / f"dockq_ost_{pred_dir.name}_all_samples.csv"

    pdb_to_ifaces = load_interfaces()
    n_ifaces = sum(len(v) for v in pdb_to_ifaces.values())
    print(f"Structures: {len(pdb_to_ifaces)}  Interfaces: {n_ifaces}")

    done = load_done(out_csv)
    print(f"Already done: {len(done)}")

    tasks = []
    for pdb_base, iface_entries in pdb_to_ifaces.items():
        pred_pdb_dir = pred_dir / pdb_base
        if not pred_pdb_dir.exists():
            continue

        # All interfaces for this structure share the same native CIF
        native_cif = iface_entries[0][3]
        iface_list = [(aid, c1, c2) for aid, c1, c2, _ in iface_entries]

        seed_dirs = []
        for p in pred_pdb_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name == "baseline_seed":
                seed_dirs.append(("baseline_seed", p))
            elif p.name.startswith("seed_"):
                seed_dirs.append((p.name.split("_")[1], p))
        seed_dirs.sort(key=lambda x: -1 if x[0] == "baseline_seed" else int(x[0]))
        for seed, seed_dir in seed_dirs:
            pred_subdir = seed_dir / "predictions"
            conf_files  = sorted(pred_subdir.glob(f"{pdb_base}_summary_confidence_sample_*.json"))

            for conf_file in conf_files:
                sample = int(conf_file.stem.split("_sample_")[1])

                # Skip only if ALL interfaces for this group are already done
                if all((aid, c1, c2, str(seed), sample) in done for aid, c1, c2 in iface_list):
                    continue

                cif = conf_file.with_name(
                    conf_file.name.replace("_summary_confidence", "").replace(".json", ".cif")
                )
                if not cif.exists():
                    continue

                try:
                    conf = json.loads(conf_file.read_text())
                except Exception:
                    conf = {}

                tasks.append((pdb_base, seed, sample, cif, native_cif, conf, iface_list))

    rows_per_group = len(next(iter(pdb_to_ifaces.values()))) if pdb_to_ifaces else 0
    print(f"OST tasks to run: {len(tasks)}  (each produces {rows_per_group}+ rows)")
    if not tasks:
        print("Nothing to do.")
        return

    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    write_lock   = threading.Lock()

    with open(out_csv, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(evaluate_group, t): t for t in tasks}
            with tqdm(total=len(tasks), unit="OST run", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        rows = future.result()
                        # A group re-runs if ANY of its interfaces is missing, but
                        # only the missing interfaces should be written — drop rows
                        # already in the CSV so resume never duplicates.
                        rows = [
                            r for r in rows
                            if (r["assembly_id"], r["chain1"], r["chain2"],
                                str(r["seed"]), int(r["sample"])) not in done
                        ]
                        if rows:
                            with write_lock:
                                writer.writerows(rows)
                                csvfile.flush()
                    except Exception as e:
                        tqdm.write(f"ERROR: {e}")
                    finally:
                        pbar.update(1)

    print(f"\nDone. Written to {out_csv}")


if __name__ == "__main__":
    main()
