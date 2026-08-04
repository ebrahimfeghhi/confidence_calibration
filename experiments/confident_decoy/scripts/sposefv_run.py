import subprocess,time,os,json,glob
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
PY="/home/ebrahim/Protenix/.venv/bin/python"; PROT="/home/ebrahim/Protenix"
OUT=f"{SC}/sposefv_out"; SENT=f"{SC}/EXPERIMENTS_RUNNING"; STATUS=f"{SC}/sposefv_status.json"
open(SENT,"w").write("sposefv")
GENENV=dict(os.environ); GENENV.update(LAYERNORM_TYPE="openfold",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
def log(m): json.dump({"t":time.strftime("%H:%M:%S"),"msg":m},open(STATUS,"w")); print(f"{time.strftime('%H:%M:%S')} {m}",flush=True)
def inf(js,seeds,gpu,lf):
    env=dict(GENENV); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    return subprocess.Popen([PY,"runner/inference.py","--model_name","protenix-v2","--load_checkpoint_dir","/home/ebrahim/checkpoint",
        "--input_json_path",js,"--dump_dir",OUT,"--seeds",seeds,"--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200","--use_template","true"],
        cwd=PROT,env=env,stdout=open(lf,"w"),stderr=subprocess.STDOUT)
jobs=json.load(open(f"{SC}/sposefv_decoys_jobs.json"))
j11=[x for x in jobs if x["pdb"]!="7z2m"]; json.dump(j11,open(f"{SC}/sposefv_no7z2m_jobs.json","w"))
log("gen 11 pdbs via gen_generic on GPUs 0-3")
subprocess.run([PY,f"{SC}/gen_generic.py",f"{SC}/sposefv_no7z2m_jobs.json",OUT],cwd=PROT,stdout=open(f"{SC}/sposefv_gen11.log","w"),stderr=subprocess.STDOUT)
log("gen 7z2m 4-way split across GPUs 0-3")
seeds=list(range(101,151)); chunks=[seeds[i::4] for i in range(4)]
ps=[inf(f"{SC}/sposefv_tmpl/7z2m-sposefv.json",",".join(map(str,chunks[g])),g,f"{SC}/sposefv_7z2m_g{g}.log") for g in range(4)]
for p in ps: p.wait()
# wait all 12 pdbs have 50
DECOYS=list(json.load(open(f"{SC}/agonly_antigen_idx.json")))
t0=time.time()
while time.time()-t0<3600:
    if all(len(glob.glob(f"{OUT}/{p}_sposefv/seed_*"))>=50 for p in DECOYS): break
    time.sleep(30)
log("scoring")
subprocess.run(["python3",f"{SC}/score_generic.py",OUT,"decoys","sposefv",f"{SC}/sposefv_scores.json"],stdout=open(f"{SC}/sposefv_score.log","w"),stderr=subprocess.STDOUT)
try: summ=json.load(open(f"{SC}/sposefv_scores_summary.json"))
except: summ=None
log(f"DONE result={summ}")
if os.path.exists(SENT): os.remove(SENT)
