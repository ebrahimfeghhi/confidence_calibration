import json,sys
from ost import io
from ost.mol.alg import lddt
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
TLC={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
MANIFEST=sys.argv[1] if len(sys.argv)>1 else f"{SC}/paratope_manifest.json"
OUTJ=sys.argv[2] if len(sys.argv)>2 else f"{SC}/paratope_lddt.json"
man=json.load(open(MANIFEST))
out=[]
for e in man:
    try:
        ref=io.LoadPDB(e["native"]).Select('peptide=true')
        mdl=io.LoadPDB(e["model"]).Select('peptide=true')
        sc=lddt.lDDTScorer(ref)
        try: g,_=sc.lDDT(mdl,local_lddt_prop='lddt')
        except: g=sc.lDDT(mdl,local_lddt_prop='lddt')
    except Exception as ex:
        out.append(dict(pdb=e["pdb"],qi=e["qi"],tag=e["tag"],err=str(ex))); continue
    mseq="";scores=[]
    for r in mdl.residues:
        mseq+=TLC.get(r.name,'X'); scores.append(r.GetFloatProp('lddt') if r.HasProp('lddt') else None)
    abseq=e.get("ab_seq",""); para=set(e["paratope"])
    off=mseq.find(abseq) if abseq else -1
    if off<0 and len(mseq)==len(abseq): off=0
    ps=[]
    if off>=0:
        for j in para:
            mi=off+j
            if 0<=mi<len(scores) and scores[mi] is not None: ps.append(scores[mi])
    out.append(dict(pdb=e["pdb"],qi=e["qi"],tag=e["tag"],npar=e["npar"],
        glob=round(g,4), para=round(sum(ps)/len(ps),4) if ps else None, npar_matched=len(ps), off=off))
json.dump(out,open(OUTJ,"w"),indent=1)
print("S2_DONE",len(out))
