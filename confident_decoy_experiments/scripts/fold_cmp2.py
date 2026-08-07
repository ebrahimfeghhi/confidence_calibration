import gemmi,subprocess,os,json,glob,re,statistics,difflib
from concurrent.futures import ThreadPoolExecutor
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
OST="/home/ebrahim/miniconda3/envs/foldbench/bin/ost"
PRED="/home/ebrahim/Protenix/foldbench_abag/predictions_template"; GT="/home/ebrahim/FoldBench/examples/ground_truths"
env=dict(os.environ); env["LD_LIBRARY_PATH"]="/home/ebrahim/miniconda3/envs/foldbench/lib"
os.makedirs(f"{SC}/native_chains2",exist_ok=True); os.makedirs(f"{SC}/lddt_out2",exist_ok=True); os.makedirs(f"{SC}/tmpc",exist_ok=True)
meta=json.load(open(f"{SC}/meta.json"))
mono={}
for g in range(4):
    for o in json.load(open(f"{SC}/mono_g{g}.json")): mono[o['name']]=o
def seqof(ch): return gemmi.one_letter_code([r.name for r in ch]).replace('X','')
def kind(s):
    hasW=re.search(r'WG.G[TS]',s);hasF=re.search(r'FG.G[TS]',s);w=s[:24]
    ntL=re.search(r'([DE]I[QV][ML]TQ|[DE]VVMTQ|EIVLTQ|QSVLTQ|DIQLTQ|SYELTQ|DIVMTQ)',w)
    ntH=re.search(r'([QE][VL][QK]L[VLEQ][EQ]S|QVKLEES|VESGG[GA]|VQSGAE|VETGGG|QESGGG)',w)
    if not (95<=len(s)<=275): return 'ANTIGEN'
    light=bool(ntL) or (bool(hasF) and not hasW); heavy=bool(ntH) or bool(hasW)
    if not(light or heavy): return 'ANTIGEN'
    if light and not ntH: return 'LIGHT'
    return 'HEAVY'
# native ref by sequence (once per (pdb,role))
_natcache={}
def native_by_seq(pdb,qseq,tag):
    if tag in _natcache: return _natcache[tag]
    p=f"{SC}/native_chains2/{tag}.pdb"
    if not os.path.exists(p):
        s=gemmi.read_structure(f"{GT}/{pdb}-assembly1.cif")
        best=None;br=-1
        for ch in s[0]:
            r=difflib.SequenceMatcher(None,seqof(ch),qseq,autojunk=False).ratio()
            if r>br: br=r;best=ch
        st=gemmi.Structure();st.add_model(gemmi.Model('1'));st[0].add_chain(best);st.setup_entities();st.write_pdb(p)
    _natcache[tag]=p; return p
# build targets
pdbs={}
for ifc,d in meta.items():
    pdb=d['pdb']
    if pdb in pdbs: continue
    cm={c['qidx']:c['native_letter'] for c in d['chain_map']}
    chs={q:mono[f'{pdb}_mono_q{q}']['sequences'][0]['proteinChain']['sequence'] for q in cm}
    kinds={q:kind(s) for q,s in chs.items()}
    ags=[q for q,k in kinds.items() if k=='ANTIGEN']; lights=[q for q,k in kinds.items() if k=='LIGHT']; heavies=[q for q,k in kinds.items() if k=='HEAVY']
    tg={'antigen':[(q,chs[q]) for q in ags],'ab':[]}
    if lights:
        for q in heavies: tg['ab'].append(('H',q,chs[q],f'{pdb}_abpair'))
        for q in lights:  tg['ab'].append(('L',q,chs[q],f'{pdb}_abpair'))
    else:
        for q in heavies: tg['ab'].append(('VHH',q,chs[q],f'{pdb}_mono_q{q}'))
    pdbs[pdb]=tg
def iso_poses(name): return sorted(glob.glob(f"{SC}/mono50_out/{name}/seed_*/predictions/{name}_sample_*.cif"))
def cplx_poses(pdb): return sorted(glob.glob(f"{PRED}/{pdb}/seed_*/predictions/{pdb}_sample_*.cif"))
def lddt(m,r,tag):
    o=f"{SC}/lddt_out2/{tag}.json"
    if not(os.path.exists(o) and os.path.getsize(o)>0):
        subprocess.run(["taskset","-c","0-79",OST,"compare-structures","-m",m,"-r",r,"-o",o,"--fault-tolerant","--min-pep-length","4","--min-nuc-length","4","--lddt"],env=env,capture_output=True,timeout=180)
    try: return json.load(open(o)).get("lddt")
    except: return None
def score(a):
    role,cif,nref,qseq,tag=a
    try: s=gemmi.read_structure(cif)
    except: return role,None,None
    best=None;br=-1
    for ch in s[0]:
        r=difflib.SequenceMatcher(None,seqof(ch),qseq,autojunk=False).ratio()
        if r>br: br=r;best=ch
    if best is None: return role,None,None
    pl=statistics.mean([at.b_iso for rr in best for at in rr])/100.0
    st=gemmi.Structure();st.add_model(gemmi.Model('1'));st[0].add_chain(best);st.setup_entities()
    mp=f"{SC}/tmpc/{tag}.pdb"; st.write_pdb(mp)
    ld=lddt(mp,nref,tag)
    try: os.remove(mp)
    except: pass
    return role,ld,pl
jobs=[]
for pdb,tg in pdbs.items():
    for q,seq in tg['antigen']:
        nref=native_by_seq(pdb,seq,f'{pdb}_ag')
        for i,c in enumerate(iso_poses(f'{pdb}_mono_q{q}')): jobs.append((f'{pdb}|ag|iso',c,nref,seq,f'{pdb}_agiso_{i}'))
        for i,c in enumerate(cplx_poses(pdb)):               jobs.append((f'{pdb}|ag|cplx',c,nref,seq,f'{pdb}_agcplx_{i}'))
    for role,q,seq,iso in tg['ab']:
        nref=native_by_seq(pdb,seq,f'{pdb}_ab{role}')
        for i,c in enumerate(iso_poses(iso)):  jobs.append((f'{pdb}|ab{role}|iso',c,nref,seq,f'{pdb}_ab{role}iso_{i}'))
        for i,c in enumerate(cplx_poses(pdb)): jobs.append((f'{pdb}|ab{role}|cplx',c,nref,seq,f'{pdb}_ab{role}cplx_{i}'))
print("tasks:",len(jobs),flush=True)
res={};done=[0]
def run(a):
    r=score(a); done[0]+=1
    if done[0]%2000==0: print(f'  {done[0]}/{len(jobs)}',flush=True)
    return (a[0],r[1],r[2])
with ThreadPoolExecutor(max_workers=24) as ex:
    for key,ld,pl in ex.map(run,jobs): res.setdefault(key,[]).append((ld,pl))
json.dump(res,open(f"{SC}/fold_cmp2_raw.json","w")); print("DONE_SCORE2",flush=True)
