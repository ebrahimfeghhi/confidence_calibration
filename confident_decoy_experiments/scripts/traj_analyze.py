#!/usr/bin/env python
"""Per-recycle epitope trajectory for one complex, from the decode-at-every-recycle dumps.

For each seed and recycle we read the DECODED structure (not the distogram, which turned out to be a
hedged proxy: AUC 0.90 for the epitope region but only +0.06 correlation with where the pose lands)
and take the true epitope -- antigen residues within 5 A of the antibody.

Reported per interface:
  consec   mean jaccard between consecutive recycles      (low = the epitope moves every cycle)
  to_final mean jaccard of each recycle to the seed's own final epitope
  best_nat best jaccard to the CRYSTAL epitope reached at ANY recycle
  end_nat  jaccard to the crystal epitope at the final recycle
The gap between best_nat and end_nat is the interesting one: it says the trajectory visited the right
site and then left it.

Antigen/antibody model chains come from OST's own mapping for a scored pose of the same complex, so
they cannot be mismatched; model chains are named in asym order, so chain letter -> asym index.
"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, "scripts")
from scipy.spatial import cKDTree
from cdr_h3_rmsd import GT_DIR, _ost, align_map, chain_residues
from epitope_rmsd import native_epitope
from iface_chains import first_pose_with

CUT = 5.0


def chain_token_base(f, ag_asym):
    """Lowest TOKEN index belonging to the antigen chain -- the chain's start, not the epitope's.

    Using the lowest epitope token instead shifted every residue number and made a solved interface
    report zero overlap with the crystal epitope.
    """
    z = np.load(f)
    tok = np.unique(z["atom_to_token_idx"][z["asym_id"][z["atom_to_token_idx"]] == ag_asym])
    return int(tok.min()), int(tok.max())


def epitope_from_coords(f, ag_asym, ab_asyms):
    """-> list of epitopes, one per diffusion sample stored for this recycle.

    coords is [N_sample, N_atom, 3] since DUMP_CYCLE_NSAMPLE was introduced (older files are
    [N_atom, 3] and are treated as a single sample).
    """
    z = np.load(f)
    xyz, a2t, asym = z["coords"], z["atom_to_token_idx"], z["asym_id"]
    if xyz.ndim == 2:
        xyz = xyz[None]
    at = asym[a2t]
    ag, ab = at == ag_asym, np.isin(at, ab_asyms)
    if not ag.any() or not ab.any():
        return [set()]
    tok = a2t[ag]
    out = []
    for x in xyz:
        hit = cKDTree(x[ab]).query_ball_point(x[ag], CUT)
        out.append(set(np.unique(tok[[len(h) > 0 for h in hit]]).tolist()))
    return out


def analyse(pdb, ifaces, dump_dir=None, seed_list=None, min_files=80):
    d = dump_dir or f"artifacts/traj_cycles/{pdb}"
    files = sorted(glob.glob(f"{d}/*_coords.npz"))
    if len(files) < min_files:
        return None
    seeds = seed_list or json.load(open("artifacts/traj20_seeds.json"))[pdb]["seeds"]
    per = {}
    for f in files:
        m = re.search(r"_pass(\d+)_cycle(\d+)_coords", os.path.basename(f))
        per.setdefault(int(m.group(1)), {})[int(m.group(2))] = f
    _, io, _ = _ost()
    ref = io.LoadMMCIF(f"{GT_DIR}/{pdb}-assembly1.cif", fault_tolerant=True)
    out = []
    for iface in ifaces:
        ab_l, ag_l = iface.rsplit("_", 1)[1][0], iface.rsplit("_", 1)[1][1]
        try:
            ch, s0, k0 = first_pose_with(pdb, ab_l, ag_l)
        except KeyError:
            continue
        ci = lambda c: ord(c.split("-")[0][0].upper()) - 65
        ag_asym = ci(ch["mdl_ag"])
        ab_asyms = [ci(ch["mdl_ab"])]
        nat_model = native_epitope(ref, ch["ref_ab"], ch["ref_ag"])
        # crystal epitope in GT numbering -> compare via GT numbering by mapping model tokens
        rres = chain_residues(ref, ch["ref_ag"])
        # model antigen sequence comes from the input json entity; align it to the GT chain
        import json as _j
        _d = _j.load(open(f"/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json"))
        _b = _d[0] if isinstance(_d, list) else _d
        _seqs = [v["sequence"] for x in _b["sequences"] for t, v in x.items()
                 if t == "proteinChain" for _ in range(int(v.get("count", 1)))]
        _mseq = _seqs[ag_asym] if ag_asym < len(_seqs) else ""
        amap = align_map(_mseq, "".join(x[1] for x in rres)) if _mseq else {}
        J = lambda a, b: len(a & b) / len(a | b) if (a | b) else 1.0
        rows = []
        for p, cyc in sorted(per.items()):
            if len(cyc) < 10:
                continue
            allsamp = {k: epitope_from_coords(cyc[k], ag_asym, ab_asyms) for k in sorted(cyc)}
            ks = sorted(allsamp)
            # representative epitope per recycle = the sample closest to the crystal, so the metric
            # is "did this trajectory reach the site at this recycle" rather than a coin flip over
            # which of the 5 diffusion draws we happened to keep
            eps = {k: v[0] for k, v in allsamp.items()}
            # token -> position in the MODEL antigen chain -> GT residue number by sequence
            # alignment (the crystal usually starts a few residues in, so positions are not 1:1).
            base, _ = chain_token_base(cyc[ks[0]], ag_asym)
            gt = lambda s: {rres[amap[x - base]][0] for x in s
                            if (x - base) in amap}
            consec = np.mean([J(eps[ks[i]], eps[ks[i + 1]]) for i in range(len(ks) - 1)])
            tofin = np.mean([J(eps[k], eps[ks[-1]]) for k in ks[:-1]])
            nat = [max(J(gt(e), nat_model) for e in allsamp[k]) for k in ks]
            # spread among samples of the SAME recycle = diffusion noise, for comparison with the
            # change BETWEEN recycles
            spread_trace = [float(np.mean([1 - J(a, b) for i, a in enumerate(allsamp[k])
                                           for b in allsamp[k][i + 1:]] or [np.nan])) for k in ks]
            samp_spread = float(np.nanmean(spread_trace)) if len(allsamp[ks[0]]) > 1 else float("nan")
            # ACROSS-loop movement measured the same way as the within-loop spread: every sample at
            # recycle k against every sample at k+1. Pairing sample slot i to slot i (as `consec`
            # does) is arbitrary -- the slots are independent draws, not a correspondence -- so this
            # is the version that is directly comparable to spread_trace.
            move_trace = [float(np.mean([1 - J(a, b) for a in allsamp[ks[i]]
                                         for b in allsamp[ks[i + 1]]]))
                          for i in range(len(ks) - 1)]
            rows.append((seeds[p] if p < len(seeds) else p, consec, tofin, max(nat), nat[-1],
                         [round(float(x), 3) for x in nat], samp_spread,
                         [None if np.isnan(x) else round(x, 3) for x in spread_trace],
                         [round(x, 3) for x in move_trace]))
        if rows:
            a = np.array([r[1:5] for r in rows])
            spread = float(np.nanmean([r[6] for r in rows]))
            # per-seed values matter more than the mean: these seeds were chosen to be DIVERSE, so
            # most of them are deliberately not the good pose and the mean is dragged down.
            out.append(dict(interface=iface, n_seeds=len(rows),
                            consec=float(a[:, 0].mean()), to_final=float(a[:, 1].mean()),
                            best_nat=float(a[:, 2].mean()), end_nat=float(a[:, 3].mean()),
                            visits_lost=float((a[:, 2] - a[:, 3]).mean()),
                            max_best_nat=float(a[:, 2].max()), max_end_nat=float(a[:, 3].max()),
                            n_seeds_visit_native=int((a[:, 2] >= 0.5).sum()),
                            n_seeds_end_native=int((a[:, 3] >= 0.5).sum()),
                            per_seed=[[int(r[0]), round(r[1], 3), round(r[2], 3),
                                       round(r[3], 3), round(r[4], 3)] for r in rows],
                            nat_trace={int(r[0]): r[5] for r in rows},
                            sample_spread=spread,
                            spread_trace={int(r[0]): r[7] for r in rows},
                            move_trace={int(r[0]): r[8] for r in rows}))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="dump dir override (default artifacts/traj_cycles/<pdb>)")
    ap.add_argument("--pdb", help="single pdb to analyse; required with --dir")
    ap.add_argument("--seeds", help="comma-separated seed order matching pass00,pass01,...")
    ap.add_argument("--min-files", type=int, default=80)
    ap.add_argument("--out", default="artifacts/traj_results.json")
    A = ap.parse_args()
    if A.dir:
        ifs = [i for i in json.load(open("artifacts/traj20.json"))
               if i.rsplit("_", 1)[0] == A.pdb]
        r = analyse(A.pdb, ifs, A.dir,
                    [int(x) for x in A.seeds.split(",")] if A.seeds else None,
                    A.min_files) or []
        for x in r:
            print(f'{x["interface"]:<12} n={x["n_seeds"]:>2}  consec {x["consec"]:.2f}  '
                  f'| best_nat max {x["max_best_nat"]:.2f}  end_nat max {x["max_end_nat"]:.2f}  '
                  f'| within-recycle spread {x.get("sample_spread", float("nan")):.2f}')
            for s, tr in x["nat_trace"].items():
                print(f'   seed {s}: ' + " ".join(f"{v:.2f}" for v in tr))
        json.dump(r, open(A.out, "w"), indent=1)
        print(f"wrote {A.out}")
        return
    ifmap = {}
    for i in json.load(open("artifacts/traj20.json")):
        ifmap.setdefault(i.rsplit("_", 1)[0], []).append(i)
    done = {}
    if os.path.exists("artifacts/traj_results.json"):
        done = {r["interface"]: r for r in json.load(open("artifacts/traj_results.json"))}
    for pdb, ifs in ifmap.items():
        if all(i in done for i in ifs):
            continue
        r = analyse(pdb, ifs)
        if not r:
            continue
        for x in r:
            done[x["interface"]] = x
            print(f'{x["interface"]:<12} n={x["n_seeds"]:>2}  consec {x["consec"]:.2f}  '
                  f'| best_nat max {x["max_best_nat"]:.2f}  end_nat max {x["max_end_nat"]:.2f}  '
                  f'| seeds VISITING native {x["n_seeds_visit_native"]}/{x["n_seeds"]}  '
                  f'ENDING native {x["n_seeds_end_native"]}/{x["n_seeds"]}  '
                  f'| within-recycle spread {x.get("sample_spread", float("nan")):.2f}', flush=True)
    json.dump(list(done.values()), open("artifacts/traj_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
