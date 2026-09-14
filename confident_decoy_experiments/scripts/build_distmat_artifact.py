#!/usr/bin/env python
"""Per-interface epitope distance-matrix heatmaps -> one browsable HTML artifact.

Each panel is the pose-vs-pose Jaccard distance matrix for one interface, reordered by average
linkage (ordering only -- the evidence is the distances). Blocks with sharp edges mean the model
samples a few DISCRETE antigen surfaces; a smooth gradient means a continuum.

MODE COUNT uses ONE GLOBAL RULE for every interface, never per-interface tuning: cut the dendrogram
at 0.9 and count clusters holding >=10% of the poses. Whether that count is STABLE across cuts
(0.75 / 0.90 / 0.95) is reported separately -- a stable count means the interface really is discrete
and the number is meaningful; an unstable one means no cluster number describes it, which is itself
the finding rather than something to tune away.
"""
import base64
import csv
import glob
import io as _io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
CUT, FLOOR = 0.9, 0.10


def modes(Z, n, cut, floor=FLOOR):
    lab = fcluster(Z, t=cut, criterion="distance")
    s = np.bincount(lab)[1:]
    return int((s >= floor * n).sum()), lab


def panel(iface, npz):
    z = np.load(npz, allow_pickle=True)
    D, dq, jn = z["D"], z["dockq"], z["jac_native"]
    # NON-CONTACTING poses (antibody touches the antigen nowhere) are NOT a surface. The Jaccard
    # distance between two empty sets falls through to 0, which made them look like one perfectly
    # identical cluster -- a solid black block that got counted as a discrete mode. 3.8% of poses
    # overall, but 64% on 8dn7_FA. Drop them here and report the count separately.
    keep = np.array([bool(x) for x in z["sets"]])
    n_empty = int((~keep).sum())
    D, dq, jn = D[np.ix_(keep, keep)], dq[keep], jn[keep]
    n = len(dq)
    if n < 50:
        return None
    Z = linkage(squareform(D, checks=False), method="average")
    k, lab = modes(Z, n, CUT)
    # ORDER POSES BY CLOSENESS TO THE CRYSTAL EPITOPE, best first. Linkage ordering is arbitrary
    # about which block lands top-left, so you cannot tell by eye whether the big block is the right
    # surface or a decoy. Sorting by jaccard-to-native puts the good poses in the top-left corner, so
    # a dark square there means the near-native poses AGREE with each other (one real basin) while a
    # mottled corner means "near-native" poses are themselves heterogeneous.
    order = np.argsort(-jn, kind="stable")
    stable = len({modes(Z, n, c)[0] for c in (0.75, 0.90, 0.95)}) == 1
    big = [c for c in range(1, lab.max() + 1) if (lab == c).sum() >= FLOOR * n]
    best = max(big, key=lambda c: jn[lab == c].mean()) if big else None
    corr_share = float((lab == best).mean()) if best is not None else 0.0
    corr_jn = float(jn[lab == best].mean()) if best is not None else 0.0
    has_correct = corr_jn >= 0.30

    # Histogram above the matrix: how close to the CRYSTAL epitope each pose actually is. The matrix
    # alone only shows poses relative to EACH OTHER, so a single tight cluster looks identical
    # whether it sits on the native epitope or on a decoy surface. This panel supplies the missing
    # absolute reference.
    fig = plt.figure(figsize=(5.0, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.1], width_ratios=[24, 1],
                          hspace=0.42, wspace=0.06)
    hx = fig.add_subplot(gs[0, 0])
    hx.hist(jn, bins=np.linspace(0, 1, 26), color="#a8432a", alpha=0.85)
    hx.axvline(float(np.median(jn)), color="#2f6d54", lw=1.8, ls="--",
               label=f"median {np.median(jn):.2f}")
    hx.set_xlim(0, 1)
    hx.set_xlabel("epitope Jaccard to the crystal epitope", fontsize=8, labelpad=3)
    hx.set_ylabel("poses", fontsize=8)
    hx.tick_params(labelsize=7)
    hx.legend(frameon=False, fontsize=7, loc="upper right")

    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(D[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    # where along the axis the poses stop being native-like
    for thr, c in ((0.5, "#2f6d54"), (0.25, "#9a7326")):
        m = int((jn[order] >= thr).sum())
        if 0 < m < n:
            ax.axhline(m - 0.5, color=c, lw=1.1, ls="--")
            ax.axvline(m - 0.5, color=c, lw=1.1, ls="--")
    ax.set_xlabel("pose, sorted by Jaccard to native (best first)", fontsize=9, labelpad=6)
    ax.set_ylabel("pose, sorted by Jaccard to native", fontsize=9, labelpad=6)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pose-vs-pose Jaccard distance", fontsize=8, labelpad=8)
    cb.ax.tick_params(labelsize=7)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=118, bbox_inches="tight")
    plt.close(fig)
    return dict(interface=iface, n=n, n_empty=n_empty, k=k, stable=bool(stable),
                dq_min=float(dq.min()), dq_max=float(dq.max()), dq_mean=float(dq.mean()),
                correct_share=corr_share, correct_jn=corr_jn, has_correct=bool(has_correct),
                acceptable=float((dq >= 0.23).mean()),
                png=base64.b64encode(buf.getvalue()).decode())


def main(only=None):
    """only: iterable of interface names to restrict to (default all 148)."""
    rows = []
    for f in sorted(glob.glob(f"{EXP}/artifacts/distmat/epitope_distmat_*.npz")):
        iface = os.path.basename(f)[len("epitope_distmat_"):-4]
        if only is not None and iface not in only:
            continue
        try:
            p = panel(iface, f)
        except Exception as e:
            print(f"  {iface}: {type(e).__name__}: {e}", flush=True)
            continue
        if p:
            rows.append(p)
    rows.sort(key=lambda r: (-r["k"], -r["correct_share"]))
    print(f"{len(rows)} interfaces")
    ks = [r["k"] for r in rows]
    print("mode-count distribution: " + ", ".join(
        f"k={k}: {ks.count(k)}" for k in sorted(set(ks))))
    print(f"stable across cuts: {sum(r['stable'] for r in rows)}/{len(rows)}")
    json.dump([{k: v for k, v in r.items() if k != 'png'} for r in rows],
              open(f"{EXP}/artifacts/distmat_summary.json", "w"), indent=1)
    return rows




HTML_HEAD = """<title>Epitope mode structure across 148 antibody-antigen interfaces</title>
<style>
:root{
  --bg:#faf8f5; --panel:#ffffff; --ink:#1d1a17; --muted:#6f665d; --line:#e4ded6;
  --accent:#a8432a; --ok:#2f6d54; --warn:#9a7326;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --ui:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#16141a;--panel:#1f1c23;--ink:#ece7e1;--muted:#9d938a;--line:#332e39;
        --accent:#e0765a;--ok:#6fc39c;--warn:#d6ab52;}
}
:root[data-theme="dark"]{--bg:#16141a;--panel:#1f1c23;--ink:#ece7e1;--muted:#9d938a;--line:#332e39;
        --accent:#e0765a;--ok:#6fc39c;--warn:#d6ab52;}
:root[data-theme="light"]{--bg:#faf8f5;--panel:#ffffff;--ink:#1d1a17;--muted:#6f665d;--line:#e4ded6;
        --accent:#a8432a;--ok:#2f6d54;--warn:#9a7326;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--ui);line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 70px}
h1{font-family:var(--sans);font-weight:600;font-size:1.85rem;margin:0 0 .3em;text-wrap:balance;
   letter-spacing:-.01em}
.lede{color:var(--muted);max-width:66ch;margin:0 0 26px;font-size:1.02rem}
.lede b{color:var(--ink);font-weight:600}
.rule{height:1px;background:var(--line);margin:26px 0}
.stats{display:flex;flex-wrap:wrap;gap:26px;margin:0 0 8px}
.stat .n{font-family:var(--mono);font-size:1.65rem;font-weight:600;font-variant-numeric:tabular-nums;
   display:block;line-height:1.1}
.stat .l{color:var(--muted);font-size:.79rem;text-transform:uppercase;letter-spacing:.07em}
.layout{display:grid;grid-template-columns:250px 1fr;gap:24px;margin-top:22px}
@media(max-width:820px){.layout{grid-template-columns:1fr}}
.side{border:1px solid var(--line);border-radius:9px;background:var(--panel);overflow:hidden;
      display:flex;flex-direction:column;max-height:74vh}
#q{border:0;border-bottom:1px solid var(--line);padding:10px 12px;font:inherit;font-size:.88rem;
   background:transparent;color:var(--ink);outline:none;width:100%}
.filters{display:flex;gap:6px;padding:8px 10px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.filters button{font:inherit;font-size:.74rem;padding:3px 9px;border-radius:999px;cursor:pointer;
   border:1px solid var(--line);background:transparent;color:var(--muted)}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.list{overflow-y:auto}
.item{padding:7px 12px;cursor:pointer;border-bottom:1px solid var(--line);display:flex;
      justify-content:space-between;gap:8px;align-items:baseline}
.item:hover{background:color-mix(in srgb,var(--accent) 8%,transparent)}
.item[aria-selected="true"]{background:color-mix(in srgb,var(--accent) 16%,transparent)}
.item .nm{font-family:var(--mono);font-size:.82rem}
.item .k{font-family:var(--mono);font-size:.74rem;color:var(--muted)}
.main{border:1px solid var(--line);border-radius:9px;background:var(--panel);padding:20px 22px}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
.head h2{font-family:var(--mono);font-size:1.15rem;margin:0}
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chip{font-size:.73rem;padding:2px 9px;border-radius:999px;border:1px solid var(--line);
      color:var(--muted);white-space:nowrap}
.chip.on{border-color:var(--ok);color:var(--ok)}
.chip.off{border-color:var(--warn);color:var(--warn)}
figure{margin:16px 0 0;text-align:center}
figure img{max-width:100%;height:auto;border-radius:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));gap:14px;margin-top:16px;
      border-top:1px solid var(--line);padding-top:14px}
.grid div span{display:block}
.grid .l{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}
.grid .v{font-family:var(--mono);font-size:1.02rem;font-variant-numeric:tabular-nums}
.nav{display:flex;gap:8px;margin-top:16px}
.nav button{font:inherit;font-size:.82rem;padding:5px 13px;border-radius:6px;cursor:pointer;
   border:1px solid var(--line);background:transparent;color:var(--ink)}
.nav button:hover{border-color:var(--accent);color:var(--accent)}
.note{color:var(--muted);font-size:.85rem;max-width:70ch;margin-top:12px}
code{font-family:var(--mono);font-size:.9em}
</style>"""


def build(rows, out_name="epitope_modes.html", subtitle=None):
    ks = [r["k"] for r in rows]
    disc = sum(1 for r in rows if r["k"] >= 2 and r["stable"])
    withc = sum(1 for r in rows if r["has_correct"])
    items, panels = [], []
    for i, r in enumerate(rows):
        items.append(
            f'<div class="item" role="option" data-i="{i}" data-k="{r["k"]}" '
            f'data-stable="{int(r["stable"])}" data-correct="{int(r["has_correct"])}" '
            f'aria-selected="{str(i == 0).lower()}">'
            f'<span class="nm">{r["interface"]}</span><span class="k">k={r["k"]}</span></div>')
        panels.append(dict(
            interface=r["interface"], n=r["n"], n_empty=r["n_empty"], k=r["k"], stable=r["stable"],
            has_correct=r["has_correct"], correct_share=round(100 * r["correct_share"]),
            correct_jn=round(r["correct_jn"], 2), acceptable=round(100 * r["acceptable"]),
            dq=f'{r["dq_min"]:.2f}–{r["dq_max"]:.2f}', png=r["png"]))
    body = f"""{HTML_HEAD}
<div class="wrap">
<h1>How many antigen surfaces does the model actually try?</h1>
<p class="lede">Each panel is one interface: the <b>pose-vs-pose epitope distance matrix</b> over 250
poses. Two poses are at distance 0 if they contact identical antigen residues and 1 if they share
none. Rows and columns are reordered by average linkage so similar poses sit together &mdash; the
ordering is cosmetic, the <b>distances</b> are the evidence. Poses where the antibody touches the
antigen nowhere are excluded &mdash; they are not a surface, and counting them produced a spurious
solid-black &ldquo;mode&rdquo;. Sharp-edged blocks mean the search picks
among a few discrete surfaces; a smooth gradient would mean a continuum.</p>
<div class="rule"></div>
<div class="stats">
  <div class="stat"><span class="n">{len(rows)}</span><span class="l">interfaces</span></div>
  <div class="stat"><span class="n">{disc}</span><span class="l">discrete &amp; stable</span></div>
  <div class="stat"><span class="n">{withc}</span><span class="l">have a correct mode</span></div>
  <div class="stat"><span class="n">{np.median(ks):.0f}</span><span class="l">median modes</span></div>
</div>
<p class="note">Mode count uses <b>one global rule</b> for every interface &mdash; cut at 0.9, count
clusters holding &ge;10% of poses &mdash; never per-interface tuning. <code>stable</code> means that
count is unchanged at cuts of 0.75, 0.90 and 0.95; where it is not, no cluster number describes the
interface, which is a result rather than something to tune away.</p>
<div class="layout">
  <aside class="side">
    <input id="q" placeholder="filter by PDB or chains…" autocomplete="off">
    <div class="filters">
      <button data-f="all" aria-pressed="true">all</button>
      <button data-f="stable" aria-pressed="false">stable</button>
      <button data-f="multi" aria-pressed="false">k&ge;2</button>
      <button data-f="nocorrect" aria-pressed="false">no correct mode</button>
    </div>
    <div class="list" id="list" role="listbox">{''.join(items)}</div>
  </aside>
  <section class="main">
    <div class="head"><h2 id="ttl"></h2><div class="chips" id="chips"></div></div>
    <figure><img id="img" alt=""></figure>
    <div class="grid" id="grid"></div>
    <div class="nav">
      <button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
      <span class="note" style="margin:0;align-self:center">or press j / k</span>
    </div>
  </section>
</div>
</div>
<script>
const P={json.dumps(panels)};
let cur=0;
const $=id=>document.getElementById(id);
function show(i){{
  const p=P[i]; cur=i;
  $('ttl').textContent=p.interface;
  $('chips').innerHTML =
    `<span class="chip">${{p.k}} mode${{p.k==1?'':'s'}}</span>`+
    `<span class="chip ${{p.stable?'on':'off'}}">${{p.stable?'stable across cuts':'unstable — no meaningful k'}}</span>`+
    `<span class="chip ${{p.has_correct?'on':'off'}}">${{p.has_correct?'has a correct mode':'no correct mode'}}</span>`;
  $('img').src='data:image/png;base64,'+p.png;
  $('img').alt='epitope distance matrix for '+p.interface;
  $('grid').innerHTML=
    `<div><span class="l">contacting poses</span><span class="v">${{p.n}}</span></div>`+
    `<div><span class="l">non-contacting</span><span class="v">${{p.n_empty}}</span></div>`+
    `<div><span class="l">modes (k)</span><span class="v">${{p.k}}</span></div>`+
    `<div><span class="l">correct mode</span><span class="v">${{p.has_correct?p.correct_share+'%':'—'}}</span></div>`+
    `<div><span class="l">jaccard to native</span><span class="v">${{p.correct_jn}}</span></div>`+
    `<div><span class="l">acceptable poses</span><span class="v">${{p.acceptable}}%</span></div>`+
    `<div><span class="l">DockQ range</span><span class="v">${{p.dq}}</span></div>`;
  document.querySelectorAll('.item').forEach(el=>
    el.setAttribute('aria-selected', el.dataset.i==i));
  const sel=document.querySelector(`.item[data-i="${{i}}"]`);
  if(sel) sel.scrollIntoView({{block:'nearest'}});
}}
function visible(){{return [...document.querySelectorAll('.item')].filter(e=>e.style.display!=='none');}}
function step(d){{
  const v=visible(); if(!v.length) return;
  let idx=v.findIndex(e=>+e.dataset.i===cur);
  idx=(idx+d+v.length)%v.length; show(+v[idx].dataset.i);
}}
document.getElementById('list').addEventListener('click',e=>{{
  const it=e.target.closest('.item'); if(it) show(+it.dataset.i);}});
$('prev').onclick=()=>step(-1); $('next').onclick=()=>step(1);
document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='j'||e.key==='ArrowDown'){{e.preventDefault();step(1);}}
  if(e.key==='k'||e.key==='ArrowUp'){{e.preventDefault();step(-1);}}}});
let filt='all';
function apply(){{
  const q=$('q').value.trim().toLowerCase();
  document.querySelectorAll('.item').forEach(el=>{{
    const p=P[+el.dataset.i];
    let ok = !q || p.interface.toLowerCase().includes(q);
    if(ok&&filt==='stable') ok=p.stable;
    if(ok&&filt==='multi') ok=p.k>=2;
    if(ok&&filt==='nocorrect') ok=!p.has_correct;
    el.style.display= ok?'':'none';}});
}}
$('q').addEventListener('input',apply);
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{
  filt=b.dataset.f;
  document.querySelectorAll('.filters button').forEach(x=>
    x.setAttribute('aria-pressed', x===b));
  apply();}});
show(0);
</script>"""
    out = f"{EXP}/artifacts/{out_name}"
    open(out, "w").write(body)
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # restrict to an interface list, one name per line
        only = [l.strip() for l in open(sys.argv[1]) if l.strip()]
        out = sys.argv[2] if len(sys.argv) > 2 else "epitope_modes_subset.html"
        build(main(only), out)
    else:
        build(main())
