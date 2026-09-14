#!/usr/bin/env python
"""Assemble the per-interface DockQ-vs-H3 panels into one self-contained HTML slideshow.

Written by a script rather than by hand because the 161 panels are embedded as base64 data URIs
(~4 MB) -- a strict CSP blocks external images, so everything must be inline.
"""
import json, html, os

summ = json.load(open("artifacts/fnat_h3_panels/summary.json")) if os.path.exists("artifacts/fnat_h3_panels/summary.json") else None
meta = json.load(open("artifacts/fnat_h3_panels/meta.json"))
order = {"no acceptable pose": 0, "confident decoy": 1, "solved": 2}
meta.sort(key=lambda m: (order[m["status"]], -m["orc"]))
counts = {k: sum(1 for m in meta if m["status"] == k) for k in order}

cards = "\n".join(
    f'<button class="row" data-i="{i}" data-status="{html.escape(m["status"])}" '
    f'data-name="{html.escape(m["k"])}">'
    f'<span class="nm">{html.escape(m["k"])}</span>'
    f'<span class="chip s{order[m["status"]]}">{html.escape(m["status"])}</span>'
    f'<span class="num">{m["sel"]:.2f}</span></button>'
    for i, m in enumerate(meta))

payload = json.dumps([{k: m[k] for k in ("k", "n", "sel", "orc", "status", "frac_h3_sub1", "med_h3", "b64")}
                      for m in meta])

HTML = """<title>fnat vs CDR H3 RMSD, per interface</title>
<style>
:root{
  --paper:#f5f6f8; --card:#ffffff; --ink:#17202e; --muted:#5d6a80; --line:#dde2ea;
  --accent:#3b5f9e; --good:#1d6b45; --warn:#8a6a12; --bad:#a8432f; --chipbg:#eef1f6;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#101419; --card:#171d25; --ink:#e6ebf2; --muted:#93a1b5; --line:#28313d;
         --accent:#7fa3dd; --good:#5fbe8f; --warn:#d9b45a; --bad:#e08a76; --chipbg:#1e2732; }
}
:root[data-theme="dark"]{ --paper:#101419; --card:#171d25; --ink:#e6ebf2; --muted:#93a1b5;
  --line:#28313d; --accent:#7fa3dd; --good:#5fbe8f; --warn:#d9b45a; --bad:#e08a76; --chipbg:#1e2732; }
:root[data-theme="light"]{ --paper:#f5f6f8; --card:#ffffff; --ink:#17202e; --muted:#5d6a80;
  --line:#dde2ea; --accent:#3b5f9e; --good:#1d6b45; --warn:#8a6a12; --bad:#a8432f; --chipbg:#eef1f6; }
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 56px}
h1{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.01em;text-wrap:balance}
.sub{color:var(--muted);margin:0 0 18px;font-size:.95rem;max-width:74ch}
.tally{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.tally span{font-size:.8rem;padding:4px 10px;border-radius:999px;background:var(--chipbg);
  border:1px solid var(--line);font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:300px 1fr;gap:20px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px}
.side{padding:10px;max-height:600px;overflow:auto}
.filter{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}
.filter button{font:inherit;font-size:.78rem;padding:4px 9px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:transparent;color:var(--muted)}
.filter button[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
input[type=search]{width:100%;font:inherit;font-size:.85rem;padding:7px 10px;margin-bottom:8px;
  border-radius:8px;border:1px solid var(--line);background:var(--paper);color:var(--ink)}
.row{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center;width:100%;
  text-align:left;font:inherit;font-size:.85rem;padding:6px 8px;border:0;border-radius:8px;
  background:transparent;color:var(--ink);cursor:pointer}
.row:hover{background:var(--chipbg)}
.row[aria-current="true"]{background:var(--accent);color:#fff}
.row[aria-current="true"] .chip,.row[aria-current="true"] .num{color:#fff}
.nm{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.chip{font-size:.68rem;padding:2px 7px;border-radius:999px;background:var(--chipbg);color:var(--muted);white-space:nowrap}
.s0{color:var(--bad)} .s1{color:var(--warn)} .s2{color:var(--good)}
.num{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.8rem;min-width:2.6em;text-align:right}
.view{padding:18px;display:flex;flex-direction:column;align-items:center;gap:14px}
.view img{max-width:100%;height:auto;border-radius:8px}
.stats{display:flex;gap:22px;flex-wrap:wrap;justify-content:center;font-size:.85rem;color:var(--muted)}
.stats b{color:var(--ink);font-variant-numeric:tabular-nums;font-weight:600}
.nav{display:flex;gap:10px;align-items:center}
.nav button{font:inherit;font-size:.85rem;padding:6px 14px;border-radius:8px;cursor:pointer;
  border:1px solid var(--line);background:var(--card);color:var(--ink)}
.nav button:hover{border-color:var(--accent);color:var(--accent)}
.nav button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.85rem;min-width:6em;text-align:center}
.legend{margin-top:22px;padding:16px 18px;border:1px solid var(--line);border-radius:12px;
  background:var(--card);font-size:.86rem;color:var(--muted);max-width:80ch}
.legend b{color:var(--ink)}
.summary{margin:0 0 22px;padding:18px;border:1px solid var(--line);border-radius:12px;background:var(--card)}
.summary h2{font-size:1.05rem;margin:0 0 6px;letter-spacing:-.01em}
.summary p{margin:0 0 14px;color:var(--muted);font-size:.88rem;max-width:78ch}
.summary img{width:100%;max-width:820px;height:auto;display:block;margin:0 auto;border-radius:8px}
.summary .keys{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px;font-size:.85rem;color:var(--muted);justify-content:center}
.summary .keys b{color:var(--ink);font-variant-numeric:tabular-nums}
.legend ul{margin:8px 0 0;padding-left:18px} .legend li{margin:4px 0}
</style>
<div class="wrap">
<h1>fnat vs CDR H3 RMSD, per interface</h1>
<p class="sub">Each panel shows the 250 poses of one antibody&ndash;antigen interface (50 seeds &times; 5 diffusion
samples). <b>fnat</b> is the fraction of native contacts recovered; <b>CDR H3 local RMSD</b> is the loop&rsquo;s own
shape. These are independent measurements &mdash; neither is used to compute the other &mdash; so unlike fnat-vs-DockQ
this comparison is not circular. Pink band: H3 below 1&nbsp;&Aring;. Green band: fnat &ge; 0.30, i.e. the epitope was
found. Gold star: the pose selected by interface pair-ipTM. Sorted hardest first.</p>
<div class="tally">
  <span>__N__ interfaces</span><span class="s0">__C0__ no acceptable pose</span>
  <span class="s1">__C1__ confident decoy</span><span class="s2">__C2__ solved</span>
</div>
__SUMMARY__
<div class="grid">
  <div class="panel side">
    <input type="search" id="q" placeholder="filter by name, e.g. 8c5h" aria-label="Filter interfaces">
    <div class="filter" id="filters">
      <button data-f="all" aria-pressed="true">all</button>
      <button data-f="no acceptable pose" aria-pressed="false">no pose</button>
      <button data-f="confident decoy" aria-pressed="false">decoy</button>
      <button data-f="solved" aria-pressed="false">solved</button>
    </div>
    <div id="list">__CARDS__</div>
  </div>
  <div class="panel view">
    <img id="img" alt="DockQ versus CDR H3 RMSD scatter for the selected interface">
    <div class="stats" id="stats"></div>
    <div class="nav">
      <button id="prev">&larr; Prev</button>
      <span class="count" id="count"></span>
      <button id="next">Next &rarr;</button>
    </div>
  </div>
</div>
<div class="legend">
  <b>How to read these.</b>
  <ul>
    <li>Points in the pink band but LOW on the y axis built the loop almost perfectly and still
        recovered almost none of the true epitope &mdash; a correct paratope on the wrong surface.</li>
    <li>The gold star is what pair-ipTM selects. A star sitting low while points sit high is a
        <i>confident decoy</i>: the right answer was generated but not chosen.</li>
    <li>Across the benchmark, the improvement in H3 from scaling explains only ~3% of the variance in
        the improvement in fnat (r = +0.19; +0.06 once fnat gain is normalised by available headroom).
        Both improve with more seeds, but largely independently.</li>
    <li>CDR H3 RMSD is <i>local</i> &mdash; the loop superposed on itself, so it measures conformation
        with placement divided out.</li>
    <li>H3 is a property of the heavy chain, so a complex with two labelled interfaces contributes the
        same H3 values to both of its panels.</li>
  </ul>
</div>
</div>
<script>
const D=__DATA__;
let cur=0, filt='all', q='';
const img=document.getElementById('img'), stats=document.getElementById('stats'),
      count=document.getElementById('count'), list=document.getElementById('list');
function visible(){return [...list.querySelectorAll('.row')].filter(r=>r.style.display!=='none');}
function show(i){
  cur=Math.max(0,Math.min(D.length-1,i)); const m=D[cur];
  img.src='data:image/png;base64,'+m.b64;
  stats.innerHTML=`<span>selected <b>${m.sel.toFixed(3)}</b></span>`+
    `<span>oracle <b>${m.orc.toFixed(3)}</b></span>`+
    `<span>poses <b>${m.n}</b></span>`+
    `<span>median H3 <b>${m.med_h3.toFixed(2)} &Aring;</b></span>`+
    `<span>H3 &le; 1 &Aring; <b>${(m.frac_h3_sub1*100).toFixed(0)}%</b></span>`;
  const vis=visible(), pos=vis.findIndex(r=>+r.dataset.i===cur);
  count.textContent=(pos<0?cur+1:pos+1)+' / '+(vis.length||D.length);
  list.querySelectorAll('.row').forEach(r=>r.setAttribute('aria-current',+r.dataset.i===cur));
  const el=list.querySelector('.row[aria-current="true"]'); if(el) el.scrollIntoView({block:'nearest'});
}
function applyFilter(){
  list.querySelectorAll('.row').forEach(r=>{
    const okS = filt==='all' || r.dataset.status===filt;
    const okQ = !q || r.dataset.name.toLowerCase().includes(q);
    r.style.display = (okS&&okQ) ? '' : 'none';
  });
  const vis=visible(); if(vis.length && !vis.some(r=>+r.dataset.i===cur)) show(+vis[0].dataset.i); else show(cur);
}
function step(d){
  const vis=visible(); if(!vis.length) return;
  let pos=vis.findIndex(r=>+r.dataset.i===cur);
  pos=(pos+d+vis.length)%vis.length; show(+vis[pos].dataset.i);
}
list.addEventListener('click',e=>{const r=e.target.closest('.row'); if(r) show(+r.dataset.i);});
document.getElementById('prev').onclick=()=>step(-1);
document.getElementById('next').onclick=()=>step(1);
document.getElementById('q').addEventListener('input',e=>{q=e.target.value.toLowerCase().trim();applyFilter();});
document.getElementById('filters').addEventListener('click',e=>{
  const b=e.target.closest('button'); if(!b) return;
  filt=b.dataset.f;
  [...e.currentTarget.children].forEach(x=>x.setAttribute('aria-pressed',x===b));
  applyFilter();
});
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='ArrowLeft'||e.key==='k'){e.preventDefault();step(-1);}
  if(e.key==='ArrowRight'||e.key==='j'){e.preventDefault();step(1);}
});
show(0);
</script>"""

SUMMARY = "" if not summ else (
 '<section class="summary"><h2>Do poses with a better loop recover more native contacts?</h2>'
 '<p>One point per interface. The horizontal axis is the Pearson correlation between CDR H3 local RMSD '
 'and <b>fnat</b> across that interface&rsquo;s 250 poses; the vertical axis is how much the loop actually varies '
 'there. Negative r means a better-built loop goes with more native contacts. Correlation is only meaningful '
 'where the loop varies &mdash; near the bottom the model builds an almost identical loop every time, so r is '
 'dominated by noise.</p>'
 '<img src="data:image/png;base64,' + summ["b64"] + '" alt="Per-interface correlation between CDR H3 RMSD and fnat">'
 '<div class="keys"><span>interfaces <b>' + str(summ["n"]) + '</b></span>'
 '<span>median r <b>' + f'{summ["median"]:+.2f}' + '</b></span>'
 '<span>negative in <b>' + str(summ["neg"]) + '</b></span>'
 '<span>|r| &gt; 0.5 in <b>' + str(summ["strong"]) + '</b></span></div></section>')

HTML = (HTML.replace("__SUMMARY__", SUMMARY).replace("__CARDS__", cards).replace("__DATA__", payload)
            .replace("__N__", str(len(meta))).replace("__C0__", str(counts["no acceptable pose"]))
            .replace("__C1__", str(counts["confident decoy"])).replace("__C2__", str(counts["solved"])))
open("artifacts/fnat_h3_slideshow.html", "w").write(HTML)
print(f"wrote artifacts/fnat_h3_slideshow.html  ({len(HTML)/1e6:.1f} MB, {len(meta)} panels)")
