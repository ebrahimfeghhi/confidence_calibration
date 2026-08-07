import subprocess,time,os,json,glob,threading,queue
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PY="/home/ebrahim/Protenix/.venv/bin/python"; PROT="/home/ebrahim/Protenix"
OUT=f"{SC}/topiptm_out"; os.makedirs(OUT,exist_ok=True)
STATUS=f"{SC}/topiptm_status.json"; GPUS=[4,5]
GENENV=dict(os.environ); GENENV.update(LAYERNORM_TYPE="openfold",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
def log(m): json.dump({"t":time.strftime("%H:%M:%S"),"msg":m},open(STATUS,"w")); print(f"{time.strftime('%H:%M:%S')} {m}",flush=True)
def inf(js,seeds,gpu,lf):
    env=dict(GENENV); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    return subprocess.Popen([PY,"runner/inference.py","--model_name","protenix-v2","--load_checkpoint_dir","/home/ebrahim/checkpoint",
        "--input_json_path",js,"--dump_dir",OUT,"--seeds",seeds,"--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200","--use_template","true"],
        cwd=PROT,env=env,stdout=open(lf,"w"),stderr=subprocess.STDOUT)
jobs=json.load(open(f"{SC}/topiptm_decoys_jobs.json"))
j11=[x for x in jobs if x["pdb"]!="7z2m"]
# 2-worker queue on GPUs 4,5 for the 11 normal pdbs
q=queue.Queue()
for x in j11: q.put(x)
def worker(gpu):
    while True:
        try: j=q.get_nowait()
        except queue.Empty: return
        p=inf(j["json"],",".join(map(str,j["seeds"])),gpu,f"{SC}/topiptm_{j['pdb']}.log"); p.wait()
        print(f"  {j['pdb']} done on GPU{gpu}",flush=True); q.task_done()
log("gen 11 pdbs on GPUs 4,5 (2 workers)")
ts=[threading.Thread(target=worker,args=(g,)) for g in GPUS]
for t in ts: t.start()
for t in ts: t.join()
# 7z2m 2-way split across GPUs 4,5
log("gen 7z2m 2-way split on GPUs 4,5")
seeds=list(range(101,151)); chunks=[seeds[i::2] for i in range(2)]
ps=[inf(f"{SC}/topiptm_tmpl/7z2m-topiptm.json",",".join(map(str,chunks[i])),GPUS[i],f"{SC}/topiptm_7z2m_g{GPUS[i]}.log") for i in range(2)]
for p in ps: p.wait()
DECOYS=list(json.load(open(f"{SC}/agonly_antigen_idx.json")))
t0=time.time()
while time.time()-t0<3600:
    if all(len(glob.glob(f"{OUT}/{p}_topiptm/seed_*"))>=50 for p in DECOYS): break
    time.sleep(30)
log("scoring")
subprocess.run(["python3",f"{SC}/score_generic.py",OUT,"decoys","topiptm",f"{SC}/topiptm_scores.json"],stdout=open(f"{SC}/topiptm_score.log","w"),stderr=subprocess.STDOUT)
try: summ=json.load(open(f"{SC}/topiptm_scores_summary.json"))
except: summ=None
log(f"DONE result={summ}")
