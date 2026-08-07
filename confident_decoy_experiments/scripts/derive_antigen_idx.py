import json,re,os
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
# port of fold_cmp.kind()
def kind(s):
    hasW=re.search(r'WG.G[TS]',s);hasF=re.search(r'FG.G[TS]',s);w=s[:24]
    ntL=re.search(r'([DE]I[QV][ML]TQ|[DE]VVMTQ|EIVLTQ|QSVLTQ|DIQLTQ|SYELTQ|DIVMTQ)',w)
    ntH=re.search(r'([QE][VL][QK]L[VLEQ][EQ]S|QVKLEES|VESGG[GA]|VQSGAE|VETGGG|QESGGG)',w)
    if not (95<=len(s)<=275): return 'ANTIGEN'
    light=bool(ntL) or (bool(hasF) and not hasW); heavy=bool(ntH) or bool(hasW)
    if not(light or heavy): return 'ANTIGEN'
    if light and not ntH: return 'LIGHT'
    return 'HEAVY'

def antigen_qidx(seqs):
    return [i for i,s in enumerate(seqs) if kind(s)=='ANTIGEN']

def seqs_from_json(path):
    o=json.load(open(path)); obj=o[0] if isinstance(o,list) else o
    return [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]

# VALIDATE on decoys against agonly_antigen_idx
known=json.load(open(f"{SC}/agonly_antigen_idx.json"))
mism=0
for pdb,aidx in known.items():
    base=f"/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json"
    got=antigen_qidx(seqs_from_json(base))
    ok = set(got)==set(aidx)
    if not ok: mism+=1; print(f"MISMATCH {pdb}: known {aidx} got {got}")
print(f"decoy validation: {len(known)-mism}/{len(known)} match")
if mism>0:
    print("ABORT: classifier disagrees with known decoy antigen idx; do NOT trust on goods"); raise SystemExit(1)

# apply to goods
safety=json.load(open(f"{SC}/safety15.json"))
good={}
for x in safety:
    pdb=x["pdb"]
    if pdb in good: continue
    base=f"/home/ebrahim/Protenix/foldbench_abag/jsons/{pdb}-final-updated.json"
    if not os.path.exists(base): print(f"MISSING base json {pdb}"); continue
    seqs=seqs_from_json(base)
    ai=antigen_qidx(seqs)
    good[pdb]=ai
    print(f"{pdb}: {len(seqs)} chains, antigen qidx={ai}, antibody kept={[i for i in range(len(seqs)) if i not in ai]}")
json.dump(good,open(f"{SC}/abonly_good_antigen_idx.json","w"))
print("GOOD_ANTIGEN_IDX_DONE",len(good))
