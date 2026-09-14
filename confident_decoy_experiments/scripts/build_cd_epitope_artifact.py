#!/usr/bin/env python
"""Per-interface view of the confident-decoy failure: the right epitope IS sampled, and not selected.

For each of the 26 confident-decoy interfaces (ipTM-selected DockQ < 0.23 while a pose >= 0.23
exists), plot every one of its ~250 poses by epitope jaccard against the crystal epitope.

  left panel  : distribution of epitope jaccard over the poses
  right panel : epitope jaccard vs the interface ipTM that does the selecting, coloured by DockQ,
                with the pose ipTM actually picks ringed

Epitope jaccard is NOT a DockQ component, so it is an independent read on whether the model found the
right surface. ipTM is the per-interface chain_pair_iptm, never the global iptm.

Writes artifacts/cd_epitopes.html.
"""
import base64, csv, glob, io as _io, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
from cdr_h3_rmsd import poses_free
from iface_chains import resolve
from selection_protocol import pair_iptm

EXP = "/home/ebrahim/tts_abag/confident_decoy_experiments"
P = list(csv.DictReader(open(f"{EXP}/artifacts/abag_selection_protocol.csv")))
CD = [r for r in P if float(r["sel50_dockq"]) < 0.23 <= float(r["oracle_dockq"])]


def build(rec):
    iface = rec["interface"]
    f = f"{EXP}/artifacts/distmat/epitope_distmat_{iface}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    keep = np.array([bool(x) for x in z["sets"]])
    jn, dq = z["jac_native"][keep], z["dockq"][keep]
    pdb, ch = iface.rsplit("_", 1)
    ab, ag = ch[0], ch[1]
    poses = [(s, k) for _, s, k, _ in poses_free(pdb) if 101 <= s <= 150]
    poses = [p for p, kp in zip(poses, keep) if kp]
    ipt = []
    for (s, k) in poses:
        try:
            c = resolve(pdb, str(s), str(k), ab, ag)
            ipt.append(pair_iptm(pdb, str(s), str(k), c["mdl_ab"], c["mdl_ag"]))
        except Exception:
            ipt.append(None)
    ipt = np.array([np.nan if v is None else v for v in ipt], float)
    if np.all(np.isnan(ipt)):
        return None
    sel = int(np.nanargmax(ipt))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    ax.hist(jn, bins=np.linspace(0, 1, 26), color="#4a7c8c", alpha=.85)
    ax.axvline(jn[sel], color="#c1553a", lw=2)
    ax.set_xlabel("epitope jaccard vs crystal", fontsize=9, labelpad=6)
    ax.set_ylabel("poses", fontsize=9)
    ax.set_title("what the model samples", fontsize=10, loc="left")
    ax.grid(alpha=.2)
    ax.text(.97, .93, f"red line = the pose ipTM picks\n({jn[sel]:.2f})", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#c1553a")
    ax = axes[1]
    s = ax.scatter(ipt, jn, c=dq, cmap="viridis", vmin=0, vmax=1, s=16,
                   edgecolor="k", linewidth=.2)
    ax.scatter([ipt[sel]], [jn[sel]], s=170, facecolor="none", edgecolor="#c1553a", lw=2, zorder=5)
    ax.set_xlabel("interface ipTM", fontsize=9, labelpad=6)
    ax.set_ylabel("epitope jaccard", fontsize=9)
    ax.set_title("what ipTM rewards", fontsize=10, loc="left")
    ax.grid(alpha=.2)
    plt.colorbar(s, ax=ax, fraction=.046, label="DockQ")
    fig.tight_layout()
    buf = _io.BytesIO(); fig.savefig(buf, format="png", dpi=118, bbox_inches="tight"); plt.close(fig)
    return dict(interface=iface, n=int(keep.sum()),
                sel_dockq=float(rec["sel50_dockq"]), oracle=float(rec["oracle_dockq"]),
                sel_jac=float(jn[sel]), best_jac=float(jn.max()),
                frac10=float((jn >= .1).mean()), frac50=float((jn >= .5).mean()),
                sel_iptm=float(ipt[sel]),
                best_jac_iptm_rank=int((ipt > ipt[int(np.argmax(jn))]).sum()) + 1,
                png=base64.b64encode(buf.getvalue()).decode())


rows = [r for r in (build(r) for r in CD) if r]
rows.sort(key=lambda r: -r["frac10"])
json.dump([{k: v for k, v in r.items() if k != "png"} for r in rows],
          open(f"{EXP}/artifacts/cd_epitopes.json", "w"), indent=1)
print(f"{len(rows)} interfaces")


HEAD = """<title>Confident decoys: the epitope is found, then discarded</title>
<style>
:root{--bg:#f7f5f2;--panel:#fff;--ink:#1b1a18;--muted:#6d665e;--line:#e3ddd5;
 --good:#2f6d54;--bad:#c1553a;--ui:system-ui,-apple-system,"Segoe UI",sans-serif;
 --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 --serif:"Iowan Old Style",Palatino,Georgia,serif;}
@media (prefers-color-scheme:dark){:root{--bg:#17161a;--panel:#201e24;--ink:#ece8e2;
 --muted:#9b938a;--line:#332f38;--good:#6fc39c;--bad:#e0765a;}}
:root[data-theme="dark"]{--bg:#17161a;--panel:#201e24;--ink:#ece8e2;--muted:#9b938a;
 --line:#332f38;--good:#6fc39c;--bad:#e0765a;}
:root[data-theme="light"]{--bg:#f7f5f2;--panel:#fff;--ink:#1b1a18;--muted:#6d665e;
 --line:#e3ddd5;--good:#2f6d54;--bad:#c1553a;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--ui);line-height:1.55}
.wrap{max-width:1120px;margin:0 auto;padding:32px 22px 64px}
h1{font-family:var(--serif);font-size:1.8rem;font-weight:600;margin:0 0 .3em;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:64ch;margin:0 0 22px}
.lede b{color:var(--ink)}
.stats{display:flex;gap:26px;flex-wrap:wrap;margin:20px 0 6px}
.stat .n{font-family:var(--mono);font-size:1.6rem;font-weight:600;display:block;line-height:1.1;
 font-variant-numeric:tabular-nums}
.stat .l{color:var(--muted);font-size:.76rem;text-transform:uppercase;letter-spacing:.07em}
.note{color:var(--muted);font-size:.85rem;max-width:70ch}
.layout{display:grid;grid-template-columns:230px 1fr;gap:22px;margin-top:20px}
@media(max-width:820px){.layout{grid-template-columns:1fr}}
.side{border:1px solid var(--line);border-radius:9px;background:var(--panel);overflow:hidden;
 display:flex;flex-direction:column;max-height:72vh}
#q{border:0;border-bottom:1px solid var(--line);padding:10px 12px;font:inherit;font-size:.86rem;
 background:transparent;color:var(--ink);outline:none}
.list{overflow-y:auto}
.item{padding:7px 12px;cursor:pointer;border-bottom:1px solid var(--line);display:flex;
 justify-content:space-between;gap:8px;align-items:baseline}
.item:hover{background:color-mix(in srgb,var(--good) 9%,transparent)}
.item[aria-selected="true"]{background:color-mix(in srgb,var(--good) 18%,transparent)}
.item .nm{font-family:var(--mono);font-size:.82rem}
.item .k{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.main{border:1px solid var(--line);border-radius:9px;background:var(--panel);padding:18px 20px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:14px;flex-wrap:wrap}
.head h2{font-family:var(--mono);font-size:1.1rem;margin:0}
figure{margin:14px 0 0;text-align:center}figure img{max-width:100%;height:auto;border-radius:5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:13px;margin-top:14px;
 border-top:1px solid var(--line);padding-top:13px}
.grid .l{color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;display:block}
.grid .v{font-family:var(--mono);font-size:1rem;font-variant-numeric:tabular-nums;display:block}
.nav{display:flex;gap:8px;margin-top:14px;align-items:center}
.nav button{font:inherit;font-size:.82rem;padding:5px 13px;border-radius:6px;cursor:pointer;
 border:1px solid var(--line);background:transparent;color:var(--ink)}
.nav button:hover{border-color:var(--good);color:var(--good)}
</style>"""


def html(rows):
    items = "".join(
        f'<div class="item" role="option" data-i="{i}" aria-selected="{str(i==0).lower()}">'
        f'<span class="nm">{r["interface"]}</span>'
        f'<span class="k">{r["frac10"]*100:.0f}%</span></div>' for i, r in enumerate(rows))
    P = [{k: v for k, v in r.items()} for r in rows]
    med = np.median([r["frac10"] for r in rows])
    body = f"""{HEAD}
<div class="wrap">
<h1>The epitope is found, then discarded</h1>
<p class="lede">All {len(rows)} <b>confident-decoy</b> interfaces: ones where ipTM selects a pose
scoring under DockQ 0.23 even though an acceptable pose exists in the same 250. Each panel plots every
pose by how much of the crystal epitope it recovers (<b>epitope jaccard</b>, which is not a DockQ
component, so it is an independent read). The ringed point is the pose ipTM picks.</p>
<div class="stats">
 <div class="stat"><span class="n">{len(rows)}</span><span class="l">interfaces</span></div>
 <div class="stat"><span class="n">{sum(1 for r in rows if r['best_jac']>=.5)}</span>
  <span class="l">reach jaccard 0.5</span></div>
 <div class="stat"><span class="n">{med*100:.0f}%</span>
  <span class="l">median poses over 0.1</span></div>
</div>
<p class="note">Sorted by the fraction of poses clearing epitope jaccard 0.1. The pattern to look for:
a healthy left histogram (the epitope is sampled) with the ring on the right panel sitting at low
jaccard (ipTM picked something else). Where the histogram is squashed near zero instead, the interface
is coverage-limited rather than selection-limited.</p>
<div class="layout">
 <aside class="side"><input id="q" placeholder="filter…" autocomplete="off">
  <div class="list" id="list" role="listbox">{items}</div></aside>
 <section class="main">
  <div class="head"><h2 id="ttl"></h2></div>
  <figure><img id="img" alt=""></figure>
  <div class="grid" id="grid"></div>
  <div class="nav"><button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
   <span class="note" style="margin:0">or j / k</span></div>
 </section>
</div></div>
<script>
const P={json.dumps(P)};let cur=0;const $=i=>document.getElementById(i);
function show(i){{const p=P[i];cur=i;$('ttl').textContent=p.interface;
 $('img').src='data:image/png;base64,'+p.png;$('img').alt='epitope jaccard for '+p.interface;
 $('grid').innerHTML=
  `<div><span class="l">poses</span><span class="v">${{p.n}}</span></div>`+
  `<div><span class="l">selected DockQ</span><span class="v">${{p.sel_dockq.toFixed(3)}}</span></div>`+
  `<div><span class="l">oracle DockQ</span><span class="v">${{p.oracle.toFixed(3)}}</span></div>`+
  `<div><span class="l">jaccard of picked</span><span class="v">${{p.sel_jac.toFixed(2)}}</span></div>`+
  `<div><span class="l">best jaccard</span><span class="v">${{p.best_jac.toFixed(2)}}</span></div>`+
  `<div><span class="l">poses &ge; 0.1</span><span class="v">${{(p.frac10*100).toFixed(0)}}%</span></div>`+
  `<div><span class="l">poses &ge; 0.5</span><span class="v">${{(p.frac50*100).toFixed(0)}}%</span></div>`+
  `<div><span class="l">ipTM of picked</span><span class="v">${{p.sel_iptm.toFixed(3)}}</span></div>`;
 document.querySelectorAll('.item').forEach(e=>e.setAttribute('aria-selected',e.dataset.i==i));
 const s=document.querySelector(`.item[data-i="${{i}}"]`);if(s)s.scrollIntoView({{block:'nearest'}});}}
function vis(){{return [...document.querySelectorAll('.item')].filter(e=>e.style.display!=='none');}}
function step(d){{const v=vis();if(!v.length)return;let i=v.findIndex(e=>+e.dataset.i===cur);
 i=(i+d+v.length)%v.length;show(+v[i].dataset.i);}}
$('list').addEventListener('click',e=>{{const it=e.target.closest('.item');if(it)show(+it.dataset.i);}});
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
document.addEventListener('keydown',e=>{{if(e.target.tagName==='INPUT')return;
 if(e.key==='j'||e.key==='ArrowDown'){{e.preventDefault();step(1);}}
 if(e.key==='k'||e.key==='ArrowUp'){{e.preventDefault();step(-1);}}}});
$('q').addEventListener('input',()=>{{const q=$('q').value.trim().toLowerCase();
 document.querySelectorAll('.item').forEach(e=>{{
  e.style.display=(!q||P[+e.dataset.i].interface.toLowerCase().includes(q))?'':'none';}});}});
show(0);
</script>"""
    open(f"{EXP}/artifacts/cd_epitopes.html", "w").write(body)
    print("wrote artifacts/cd_epitopes.html")


html(rows)
