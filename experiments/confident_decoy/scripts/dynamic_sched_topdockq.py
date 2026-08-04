import subprocess,time,os,json,glob,re
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PY="/home/ebrahim/Protenix/.venv/bin/python"; PROT="/home/ebrahim/Protenix"
OUT=f"{SC}/topdockq_out"; STATUS=f"{SC}/dynamic_sched_status.json"
GPUS=[0,1,2,3,4,5]   # 6,7 are another user's
SEEDS=list(range(101,151))
GENENV=dict(os.environ); GENENV.update(LAYERNORM_TYPE="openfold",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
jobs=json.load(open(f"{SC}/topdockq_decoys_jobs.json"))
ALL=[x["pdb"] for x in jobs]; JMAP={x["pdb"]:x["json"] for x in jobs}
def log(m): json.dump({"t":time.strftime("%H:%M:%S"),"msg":m},open(STATUS,"w")); print(f"{time.strftime('%H:%M:%S')} {m}",flush=True)
def done(pdb): return len(glob.glob(f"{OUT}/{pdb}_topdockq/seed_*"))>=50
def inflight():
    d={}
    for p in os.listdir('/proc'):
        if not p.isdigit(): continue
        try: cl=open(f'/proc/{p}/cmdline','rb').read().replace(b'\0',b' ').decode()
        except: continue
        if 'inference.py' not in cl or '-topdockq.json' not in cl: continue
        m=re.search(r'([0-9a-z]+)-topdockq\.json',cl)
        if not m: continue
        try: env=open(f'/proc/{p}/environ','rb').read().replace(b'\0',b'\n').decode()
        except: env=''
        g=re.search(r'CUDA_VISIBLE_DEVICES=(\d+)',env)
        d.setdefault(m.group(1),[]).append(int(g.group(1)) if g else -1)
    return d   # pdb -> [gpus]
def launch(pdb,seeds,gpu):
    env=dict(GENENV); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    subprocess.Popen([PY,"runner/inference.py","--model_name","protenix-v2","--load_checkpoint_dir","/home/ebrahim/checkpoint",
        "--input_json_path",JMAP[pdb],"--dump_dir",OUT,"--seeds",seeds,"--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200","--use_template","true"],
        cwd=PROT,env=env,stdout=open(f"{SC}/topdockq_{pdb}_g{gpu}.log","w"),stderr=subprocess.STDOUT)
log(f"dynamic scheduler start; pdbs={ALL}")
z_started=False
while True:
    ifl=inflight()
    busy=set(g for gs in ifl.values() for g in gs if g>=0)
    free=[g for g in GPUS if g not in busy]
    pend=[p for p in ALL if p!="7z2m" and not done(p) and p not in ifl]
    # fill free GPUs with pending non-7z2m
    for g in list(free):
        if not pend: break
        p=pend.pop(0); launch(p,",".join(map(str,SEEDS)),g); busy.add(g); free.remove(g)
        log(f"launched {p} on GPU{g}")
    # 7z2m last: when no non-7z2m pending or in-flight, split across all free GPUs
    non7_active = pend or any(p!="7z2m" and not done(p) and p in ifl for p in ALL)
    if not z_started and not done("7z2m") and "7z2m" not in ifl and not non7_active and len(free)>=1:
        nf=len(free); chunks=[SEEDS[i::nf] for i in range(nf)]
        for i,g in enumerate(free):
            if chunks[i]: launch("7z2m",",".join(map(str,chunks[i])),g)
        z_started=True; log(f"7z2m {nf}-way split across GPUs {free}")
    if all(done(p) for p in ALL): log("ALL_DONE dynamic scheduler"); break
    time.sleep(20)
