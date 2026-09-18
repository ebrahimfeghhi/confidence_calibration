#!/usr/bin/env python
"""Verify a self-template job will actually be consumed -- BEFORE spending a GPU fold on it.

WHY THIS EXISTS. Self-templated arms have failed silently or noisily in four different ways this
week: template ids colliding in the shared mmcif dir, a truncating concurrent write to the shared
release-date cache, an empty splice, and a residual failure where an arm reports "Found 0 templates"
in situ while the very same job featurises correctly in a fresh process. Every one of those wasted a
fold, and the first two produced plausible-looking results rather than errors.

This runs the REAL featuriser on the REAL job in a fresh process, on CPU, in about a second, and
reports how many templates each chain would get plus the featuriser's own errors and warnings -- the
diagnostic that the in-run path throws away.

Usage: preflight_templates.py --tmpl <dir with job.json + a3m/> --expect 5
Exit 0 if every chain yields exactly --expect templates, 1 otherwise.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/home/ebrahim/Protenix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmpl", required=True)
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--mmcif-dir", default="/home/ebrahim/mmcif")
    ap.add_argument("--release-dates", default="/home/ebrahim/common/release_date_cache.json")
    ap.add_argument("--kalign", default="/home/ebrahim/Protenix/.venv/bin/kalign")
    ap.add_argument("--quiet", action="store_true")
    A = ap.parse_args()

    from protenix.data.template.template_parser import HmmsearchA3MParser
    from protenix.data.template.template_utils import TemplateHitFeaturizer

    job_p = f"{A.tmpl}/job.json"
    if not os.path.exists(job_p):
        print(f"PREFLIGHT FAIL: no job.json at {A.tmpl}")
        return 1
    job = json.load(open(job_p))
    entry = job[0] if isinstance(job, list) else job
    seqs = [v for s in entry["sequences"] for v in s.values()
            if isinstance(v, dict) and v.get("sequence")]

    # env must match what the fold will use, or the check does not test the same thing
    os.environ.setdefault("PROTENIX_TMPL_NODEDUP", "1")
    os.environ["PROTENIX_MAX_TEMPLATES"] = str(A.expect)
    os.environ.setdefault("PROTENIX_MAX_TEMPLATE_CANDIDATES", "20")

    feat = TemplateHitFeaturizer(
        mmcif_dir=A.mmcif_dir, template_cache_dir=None, max_hits=A.expect,
        kalign_binary_path=A.kalign, max_template_date="2021-09-30",
        release_dates_path=A.release_dates, obsolete_pdbs_path=None,
        _shuffle_top_k_prefiltered=None, _max_template_candidates_num=20, fetch_remote=False)

    a3ms = sorted(glob.glob(f"{A.tmpl}/a3m/*.a3m"))
    if not a3ms:
        print(f"PREFLIGHT FAIL: no a3m files under {A.tmpl}")
        return 1
    ok = True
    for qi, a3m in enumerate(a3ms):
        if qi >= len(seqs):
            break
        seq = seqs[qi]["sequence"]
        hits = HmmsearchA3MParser.parse(query_seq=seq, a3m_str=open(a3m).read(), skip_first=False)
        res, _ = feat.get_templates(sequence_uid=seq, query_sequence=seq, hits=hits,
                                    max_template_date=None)
        n = len(res.features)
        if n != A.expect:
            ok = False
            print(f"PREFLIGHT FAIL {os.path.basename(A.tmpl)} q{qi}: "
                  f"hits {len(hits)} -> templates {n}, expected {A.expect}")
            for m in (list(res.errors) + list(res.warnings))[:3]:
                print(f"    {str(m)[:200]}")
        elif not A.quiet:
            print(f"  preflight q{qi}: {len(hits)} hits -> {n} templates OK")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
