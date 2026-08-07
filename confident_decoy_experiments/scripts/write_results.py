import json,os,time
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import EXP, CODE_ROOT
def summ(js):
    p=f"{SC}/{js}"
    if not os.path.exists(p): return None
    try: return json.load(open(p))
    except: return None
# (label, set, summary_file, config link)
RUNS=[
 ("abonly",  "decoys (15)", "abonly_scores_summary.json",        "configs/abonly_decoys.yaml", "decoy"),
 ("abonly",  "goods (15)",  "abonly_goods_scores_summary.json",  "configs/abonly_goods.yaml",  "good"),
 ("oracle",  "decoys (15)", "oracle_scores_summary.json",        "configs/oracle_decoys.yaml", "decoy"),
 ("oracle",  "goods (15)",  "oracle_goods_scores_summary.json",  "configs/oracle_goods.yaml",  "good"),
]
lines=[]
lines.append("# Confident-Decoy Rescue — Results\n")
lines.append(f"_Auto-generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Configs in `configs/`; provenance in `../confident_decoy_RUNS.md`._\n")
lines.append("## Baselines (established)")
lines.append("| method | chains templated | fold | decoys rescued | goods preserved |")
lines.append("|---|---|---|---|---|")
lines.append("| free (no template) | none | none | 0/15 | 15/15 |")
lines.append("| agonly | antigen only | native | 8/15 | 15/15 |")
lines.append("| native50 / safety | both | native | 14/15 | 15/15 |")
lines.append("| expb | both | predicted-isolation fixed | 2/15 | — |")
lines.append("| expc | both | predicted-freerun varying | 2/15 | — |")
lines.append("| expd | both | predicted-freerun best-pLDDT fixed | 0/15 | — |\n")
lines.append("## This batch (2026-07-27)")
lines.append("| run | set | result | config |")
lines.append("|---|---|---|---|")
detail=[]
for label,setname,sfile,cfg,grp in RUNS:
    s=summ(sfile)
    if not s or grp not in s:
        val="_pending_"
        lines.append(f"| **{label}** | {setname} | {val} | [{cfg}]({cfg}) |")
        continue
    metric="rescued" if grp=="decoy" else "preserved"
    val=f"**{s[grp]['count']}** {metric}"
    lines.append(f"| **{label}** | {setname} | {val} | [{cfg}]({cfg}) |")
    detail.append(f"- **{label} / {setname}** — {metric} {s[grp]['count']}: `{', '.join(s[grp]['hit'])}`")
lines.append("\n### Per-interface hits")
lines+=detail if detail else ["_none scored yet_"]
lines.append("\n## Interpretation hooks")
lines.append("- **abonly decoys vs agonly (8/15):** >8 ⇒ antibody side adds signal; ~8 or with-overlap ⇒ complementary to antigen; <8 ⇒ antigen/epitope dominant.")
lines.append("- **oracle decoys vs expd (0/15) and native (14/15):** ~14 ⇒ fold-**selection** is the wall (best generatable fold suffices, pLDDT just picks wrong); ~0–2 ⇒ **generation**-limited (even best generatable fold isn't enough, consistent with long-H3 ceiling).")
lines.append("- **goods (both runs):** must stay ~15/15 or the method breaks working cases (not deployable).")
open(f"{CODE_ROOT}/RESULTS.md","w").write("\n".join(lines)+"\n")
print("wrote",f"{CODE_ROOT}/RESULTS.md")
