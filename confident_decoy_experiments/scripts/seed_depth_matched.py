#!/usr/bin/env python
"""Seed-matched depth curve: stock (MSA subsampling + dropout) vs both-off, seeds 101-110 only.

WHY RESTRICT THE REFERENCE TO 101-110. The reference run is seeds 101-150 (250 poses) and the both-off
arm is 101-110 (50 poses), so every comparison so far confounded the two knobs with a 5x pose-count
advantage. 101-110 is a strict subset of 101-150, so cutting the reference down to those ten seeds
removes the confound entirely -- same seeds, same pose count, only the two knobs differ. No new
inference: both sides come from cached OST output.

DEPTHS
  1   each of the 10 seeds alone (5 poses)      -> 10 values, report mean +/- SD
  5   5 of the 10 drawn at random, 10 repeats   -> 10 values, report mean +/- SD
  10  all 10 seeds                              -> single value

SELECTION is the label-free rule used everywhere else: pool the subset's poses, take the single
highest interface pair_iptm pose, report ITS DockQ. Averaged over interfaces; the SD is across draws,
not across interfaces.

Only interfaces where BOTH arms have all 10 seeds scored are used, so the two columns are paired.

Usage: .../python scripts/seed_depth_matched.py
"""
import glob, json, os, random, statistics as st, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_set import GT, OST, ENV

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
A = f"{EXP}/artifacts"
PRED = "/home/ebrahim/Protenix/foldbench_abag/predictions_template"
REF_CACHE = f"{A}/abag_all_ost"
SEEDS = list(range(101, 111))
SAMPLES = range(5)
THRESH = 0.23
N_REPEAT = 10
RNG = random.Random(0)

ALL = json.load(open(f"{A}/abag_all_101_150.json"))
man = json.load(open(f"{A}/wnoise_manifest.json"))


def _pair(d, conf, c1, c2):
    """(pair_iptm, dockq) for the labelled interface, or None."""
    qs, ifs = d.get("dockq") or [], d.get("dockq_interfaces") or []
    for i, t in enumerate(ifs):
        if {t[0], t[1]} == {c1, c2} and i < len(qs):
            x, y = ord(t[2][0]) - 65, ord(t[3][0]) - 65
            if 0 <= x < len(conf) and 0 <= y < len(conf):
                return conf[x][y], qs[i]
    return None


def _subset(src, dst, keep):
    """Write `src` keeping only chains in `keep` (by the names OST/gemmi sees)."""
    import gemmi
    st = gemmi.read_structure(src)
    st.setup_entities()
    for mdl in st:
        for nm in [c.name for c in mdl]:
            if nm not in keep:
                mdl.remove_chain(nm)
    st.setup_entities()
    st.make_mmcif_document().write_file(dst)


def _isolated_pair_dockq(pdb, c1, c2, cif, tag):
    """Fallback for assemblies with duplicate copies (8fxv: A,B and A-2,B-2).

    OST's global chain mapping sometimes lands entirely on the '-2' copy, so the labelled A-B pair is
    never emitted and the pose silently disappears from the results. Mirrors _isolated_pair_dockq in
    abag_all_101_150.py: subset BOTH model and native to the two chains, then rescore.
    """
    import subprocess, tempfile
    o = f"{REF_CACHE}/iso_{tag}.json"
    if not (os.path.exists(o) and os.path.getsize(o)):
        with tempfile.TemporaryDirectory() as td:
            ms, ns = f"{td}/m.cif", f"{td}/n.cif"
            try:
                _subset(cif, ms, {c1, c2})
                _subset(f"{GT}/{pdb}-assembly1.cif", ns, {c1, c2})
            except Exception:
                return None
            subprocess.run(["taskset", "-c", "0-47", OST, "compare-structures", "-m", ms, "-r", ns,
                            "-o", o, "--fault-tolerant", "--min-pep-length", "4",
                            "--min-nuc-length", "4", "--dockq"],
                           env=ENV, capture_output=True, timeout=900)
    try:
        q = json.load(open(o)).get("dockq") or []
    except Exception:
        return None
    return q[0] if len(q) == 1 else None


def stock_poses(iface):
    """Reference arm, seeds 101-110 only, from cache with the two documented OST fallbacks."""
    import subprocess
    pdb = ALL[iface]["pdb"]; c1, c2 = ALL[iface]["chains"].split("-", 1)
    out = collections.defaultdict(list)
    for s in SEEDS:
        for k in SAMPLES:
            o = f"{REF_CACHE}/{pdb}_{s}_{k}.json"
            cf = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_summary_confidence_sample_{k}.json"
            cif = f"{PRED}/{pdb}/seed_{s}/predictions/{pdb}_sample_{k}.cif"
            if not os.path.exists(cf):
                continue
            try:
                conf = json.load(open(cf))["chain_pair_iptm"]
            except Exception:
                continue
            try:
                d = json.load(open(o))
            except Exception:
                d = {}
            # (1) chain-mapper IndexError -> status FAILURE, no dockq at all (8dtn). Retry with an
            # explicit identity map, which the mapper cannot fail on.
            if d.get("status") != "SUCCESS" and os.path.exists(cif):
                subprocess.run(["taskset", "-c", "0-47", OST, "compare-structures", "-m", cif,
                                "-r", f"{GT}/{pdb}-assembly1.cif", "-o", o, "--fault-tolerant",
                                "--min-pep-length", "4", "--min-nuc-length", "4", "--dockq",
                                "-c", f"{c1}:{c1}", f"{c2}:{c2}"],
                               env=ENV, capture_output=True, timeout=1800)
                try: d = json.load(open(o))
                except Exception: d = {}
            r = _pair(d, conf, c1, c2)
            # (2) duplicate-copy assembly -> labelled pair absent from the global mapping (8fxv).
            if r is None and os.path.exists(cif):
                dq = _isolated_pair_dockq(pdb, c1, c2, cif, f"{pdb}_{c1}{c2}_{s}_{k}")
                if dq is not None:
                    x, y = ord(c1[0]) - 65, ord(c2[0]) - 65
                    if 0 <= x < len(conf) and 0 <= y < len(conf):
                        r = (conf[x][y], dq)
            if r: out[s].append(r)
    return out


def bothoff_poses(iface):
    """sigma=0 dose arm, from its own OST cache."""
    arm = f"dose_s0_{iface}"
    c1, c2 = ALL[iface]["chains"].split("-", 1)
    out = collections.defaultdict(list)
    for cif in glob.glob(f"{A}/sweep_out/{arm}/**/*_sample_*.cif", recursive=True):
        name = os.path.basename(cif)[:-4]; s = int(cif.split("/seed_")[1].split("/")[0])
        if s not in SEEDS: continue
        o = f"{A}/{arm}_ost/{s}_{name}.json"
        cf = os.path.join(os.path.dirname(cif),
                          name.replace("_sample_", "_summary_confidence_sample_") + ".json")
        if not (os.path.exists(o) and os.path.exists(cf)): continue
        try:
            d = json.load(open(o)); conf = json.load(open(cf))["chain_pair_iptm"]
        except Exception:
            continue
        r = _pair(d, conf, c1, c2)
        if r: out[s].append(r)
    return out


def selected(poses, seeds):
    """Label-free rule: highest pair_iptm pose in the subset -> its DockQ."""
    p = [x for s in seeds for x in poses.get(s, [])]
    if not p: return None
    return max(p, key=lambda x: x[0])[1]


# ---- collect, keeping only interfaces complete on BOTH sides ------------------------------------
data = {}
for iface in sorted(ALL):
    if f"dose_s0_{iface}" not in man: continue
    b = bothoff_poses(iface)
    if len([s for s in SEEDS if b.get(s)]) < len(SEEDS): continue
    a = stock_poses(iface)
    if len([s for s in SEEDS if a.get(s)]) < len(SEEDS): continue
    data[iface] = (a, b)
ifaces = sorted(data)
print(f"paired interfaces with all 10 seeds on both arms: {len(ifaces)}\n")
if not ifaces:
    sys.exit("nothing to report yet")


def draw_stats(subsets):
    """For each seed-subset, mean selected DockQ and rescued rate over interfaces."""
    m, r = [], []
    for sub in subsets:
        v = [selected(data[i][0], sub) for i in ifaces], [selected(data[i][1], sub) for i in ifaces]
        m.append((st.mean(x for x in v[0] if x is not None),
                  st.mean(x for x in v[1] if x is not None)))
        r.append((sum(x >= THRESH for x in v[0] if x is not None) / len(ifaces),
                  sum(x >= THRESH for x in v[1] if x is not None) / len(ifaces)))
    return m, r


rows = []
for depth in (1, 5, 10):
    if depth == 1:
        subs = [[s] for s in SEEDS]
    elif depth == 5:
        subs = [RNG.sample(SEEDS, 5) for _ in range(N_REPEAT)]
    else:
        subs = [SEEDS]
    m, r = draw_stats(subs)
    f = lambda vals: (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0)
    rows.append((depth, len(subs),
                 f([x[0] for x in m]), f([x[1] for x in m]),
                 f([x[0] for x in r]), f([x[1] for x in r])))

print(f"seeds 101-110 on BOTH arms | n={len(ifaces)} interfaces | SD is across draws\n")
print(f"{'depth':>6}{'draws':>7}{'stock sel_dq':>20}{'both-off sel_dq':>20}"
      f"{'stock rescued':>18}{'both-off rescued':>19}")
print("-" * 92)
for d, nd, sm, bm, sr, br in rows:
    print(f"{d:>6}{nd:>7}{sm[0]:>13.3f} +/-{sm[1]:>5.3f}{bm[0]:>13.3f} +/-{bm[1]:>5.3f}"
          f"{sr[0]:>11.3f} +/-{sr[1]:>5.3f}{br[0]:>12.3f} +/-{br[1]:>5.3f}")
json.dump({"n_interfaces": len(ifaces), "interfaces": ifaces,
           "rows": [{"depth": d, "draws": nd, "stock_sel": sm, "bothoff_sel": bm,
                     "stock_rescued": sr, "bothoff_rescued": br} for d, nd, sm, bm, sr, br in rows]},
          open(f"{A}/seed_depth_matched.json", "w"), indent=1)
