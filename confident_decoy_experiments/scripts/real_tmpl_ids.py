#!/usr/bin/env python
"""Which 4 real templates did the FREE arm actually use for a complex?

WHY THIS EXISTS. The tmplctl run2 arm ("real templates added back alongside the 5 self-poses") has to
restore exactly the templates Arm A had. It cannot just concatenate the real hmmsearch a3m, for two
reasons:

  1. Those a3m files hold thousands of candidate hits (9nph: 5609 / 30075 / 33395). Protenix filters
     them -- dedup, then a 20-hit candidate pool (PROTENIX_MAX_TEMPLATE_CANDIDATES), then up to
     max_hits selected with release date <= 2021-09-30. Only 4 survive.
  2. run2 must set PROTENIX_TMPL_NODEDUP=1 so the 5 near-identical self-poses are not collapsed into
     one -- but that ALSO disables dedup for the real hits, so the 4 that survive would become 4
     near-duplicate chains of a single PDB entry instead of Arm A's 4 distinct structures.

So the selection is reproduced HERE, offline and CPU-only, with Arm A's settings (dedup ON, max_hits
4), and the chosen hit names are pinned. build_selftmpl_general.py --prepend-real then writes exactly
those rows, which makes the NODEDUP setting irrelevant for them.

Settings mirror protenix/data/inference/infer_dataloader.py:103 and configs/configs_data.py:281.

Usage: real_tmpl_ids.py --job artifacts/pxm2025/jsons/9nph-final-updated.json --out artifacts/tmplctl/real_tmpl.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/ebrahim/Protenix")
ROOT = os.path.expanduser("~")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="the FREE arm's job json (carries templatesPath)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hits", type=int, default=4, help="Arm A's binding cap")
    A = ap.parse_args()

    # dedup must be ON here -- this is reproducing the FREE arm, not run2
    os.environ.pop("PROTENIX_TMPL_NODEDUP", None)
    os.environ.pop("PROTENIX_MAX_TEMPLATES", None)

    from protenix.data.template.template_utils import TemplateHitFeaturizer
    from protenix.data.template.template_featurizer import HmmsearchA3MParser

    feat = TemplateHitFeaturizer(
        mmcif_dir=os.path.join(ROOT, "mmcif"),
        template_cache_dir="",
        max_hits=A.max_hits,
        kalign_binary_path=os.environ.get("KALIGN_BIN",
                                          "/home/ebrahim/Protenix/.venv/bin/kalign"),
        max_template_date="2021-09-30",
        release_dates_path=os.path.join(ROOT, "common/release_date_cache.json"),
        obsolete_pdbs_path=os.path.join(ROOT, "common/obsolete_to_successor.json"),
        _shuffle_top_k_prefiltered=None,
        _max_template_candidates_num=int(
            os.environ.get("PROTENIX_MAX_TEMPLATE_CANDIDATES", 20)),
        fetch_remote=True,
    )

    obj = json.load(open(A.job))
    obj = obj[0] if isinstance(obj, list) else obj
    out = []
    for e in obj["sequences"]:
        pc = e.get("proteinChain")
        if not pc:
            continue
        seq, tp = pc["sequence"], pc.get("templatesPath", "")
        if not tp or not os.path.exists(tp):
            print(f"  entity len {len(seq)}: no templatesPath -- 0 real templates")
            out.append(dict(sequence=seq, templates_path=tp, names=[]))
            continue
        hits = HmmsearchA3MParser.parse(query_seq=seq, a3m_str=open(tp).read(),
                                        skip_first=False)
        res, _ = feat.get_templates(sequence_uid=seq, query_sequence=seq, hits=hits,
                                    max_template_date=None)
        # get_templates returns TemplateSearchResult(features=<list of per-hit dicts>, hits=...).
        # The hit identity lives in each per-hit dict under template_domain_names; fall back to
        # res.hits if the feature key is absent, rather than silently emitting an empty list.
        names = []
        for f in (res.features or []):
            n = f.get("template_domain_names") if isinstance(f, dict) else None
            if n is None:
                continue
            # template_domain_names arrives as a 0-d object array, a 1-d array, or a bare value
            # depending on the hit -- normalise all three rather than indexing blind.
            import numpy as _np
            for v in _np.atleast_1d(_np.asarray(n, dtype=object)).ravel():
                names.append(v.decode() if isinstance(v, bytes) else str(v))
        if not names:
            for h in (getattr(res, "hits", None) or []):
                nm = getattr(h, "name", None) or getattr(h, "pdb_id", None)
                if nm:
                    names.append(nm.decode() if isinstance(nm, bytes) else str(nm))
        names = [n for n in names if n]
        print(f"  entity len {len(seq)}: {len(hits)} hits -> {len(names)} selected: {names}")
        out.append(dict(sequence=seq, templates_path=tp, names=names))

    json.dump(out, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
