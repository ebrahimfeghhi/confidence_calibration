import subprocess,time,os,json,glob
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
ESM_SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad"
PY="/home/ebrahim/Protenix/.venv/bin/python"; PROT="/home/ebrahim/Protenix"
SENTINEL=f"{SC}/EXPERIMENTS_RUNNING"; STATUS=f"{SC}/resume_status.json"
open(SENTINEL,"w").write("running")
status={"started":time.strftime("%H:%M:%S"),"stages":{}}
def save(): json.dump(status,open(STATUS,"w"),indent=1)
def log(m): print(f"{time.strftime('%H:%M:%S')} {m}",flush=True); save()
def sh(cmd,cwd,logf):
    with open(logf,"w") as f: return subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT).returncode
GENENV=dict(os.environ); GENENV.update(LAYERNORM_TYPE="openfold",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
def inf_cmd(js,out,seeds): return [PY,"runner/inference.py","--model_name","protenix-v2","--load_checkpoint_dir","/home/ebrahim/checkpoint","--input_json_path",js,"--dump_dir",out,"--seeds",seeds,"--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200","--use_template","true"]
def run_split(js,out,tag,gpus=(0,1,2,3),seeds=range(101,151)):
    seeds=list(seeds); chunks=[seeds[i::len(gpus)] for i in range(len(gpus))]; procs=[]
    for g,ch in zip(gpus,chunks):
        if not ch: continue
        env=dict(GENENV); env["CUDA_VISIBLE_DEVICES"]=str(g)
        f=open(f"{SC}/rgen_{tag}_g{g}.log","w")
        procs.append(subprocess.Popen(inf_cmd(js,out,",".join(map(str,ch))),cwd=PROT,env=env,stdout=f,stderr=subprocess.STDOUT))
    for p in procs: p.wait()
def wait_seeds(out,names,n=50,timeout=36000,poll=60):
    t0=time.time()
    while time.time()-t0<timeout:
        if all(len(glob.glob(f"{out}/{nm}/seed_*"))>=n for nm in names): return True
        time.sleep(poll)
    return False
def summ(js):
    try: return json.load(open(js.replace(".json","_summary.json")))
    except: return None
def stage(name,fn):
    status["stages"][name]={"status":"running","t":time.strftime("%H:%M:%S")}; save()
    try: status["stages"][name]={"status":"done","result":fn(),"t":time.strftime("%H:%M:%S")}
    except Exception as e: status["stages"][name]={"status":"FAILED","error":repr(e),"t":time.strftime("%H:%M:%S")}
    save(); subprocess.run(["python3",f"{SC}/write_results.py"])
DECOYS=list(json.load(open(f"{SC}/agonly_antigen_idx.json")))
GOODS=[]
for x in json.load(open(f"{SC}/safety15.json")):
    if x["pdb"] not in GOODS: GOODS.append(x["pdb"])

# R1: finish abonly 7z2m split, re-score abonly decoys accurately
def r1():
    log("wait abonly 7z2m split -> 50")
    wait_seeds(f"{SC}/abonly_out",["7z2m_abonly"],timeout=5400)
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/abonly_out","decoys","abonly",f"{SC}/abonly_scores.json"],None,f"{SC}/r_score_abonly_decoys.log")
    return summ(f"{SC}/abonly_scores.json")
stage("abonly_decoys_rescore",r1)

# R2: oracle decoys — 11 normal via gen_generic, 7z2m split 4-way, then score
def r2():
    jobs=json.load(open(f"{SC}/oracle_decoys_jobs.json"))
    j11=[x for x in jobs if x["pdb"]!="7z2m"]; json.dump(j11,open(f"{SC}/oracle_decoys_no7z2m.json","w"))
    log("oracle decoys: 11 pdbs via gen_generic")
    sh([PY,f"{SC}/gen_generic.py",f"{SC}/oracle_decoys_no7z2m.json",f"{SC}/oracle_out"],PROT,f"{SC}/r_gen_oracle_decoys11.log")
    log("oracle decoys: 7z2m 4-way split")
    run_split(f"{SC}/oracle_tmpl/7z2m-oracle.json",f"{SC}/oracle_out","oracle7z2m")
    wait_seeds(f"{SC}/oracle_out",[f"{p}_oracle" for p in DECOYS],timeout=3600)
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/oracle_out","decoys","oracle",f"{SC}/oracle_scores.json"],None,f"{SC}/r_score_oracle_decoys.log")
    return summ(f"{SC}/oracle_scores.json")
stage("oracle_decoys",r2)

# R3: abonly goods
def r3():
    log("abonly goods gen")
    sh([PY,f"{SC}/gen_generic.py",f"{SC}/abonly_goods_jobs.json",f"{SC}/abonly_out"],PROT,f"{SC}/r_gen_abonly_goods.log")
    wait_seeds(f"{SC}/abonly_out",[f"{p}_abonly" for p in GOODS],timeout=21600)
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/abonly_out","goods","abonly",f"{SC}/abonly_goods_scores.json"],None,f"{SC}/r_score_abonly_goods.log")
    return summ(f"{SC}/abonly_goods_scores.json")
stage("abonly_goods",r3)

# R4: oracle goods (PY for all build steps)
def r4():
    log("prep goods oracle")
    sh([PY,f"{SC}/prep_oracle_goods.py"],None,f"{SC}/r_prep_oracle_goods.log")
    sh([PY,f"{SC}/build_oracle_score.py",f"{SC}/oracle_chainmap_goods.json",f"{SC}/native_chains_goods",f"{SC}/oracle_folds_goods.json"],None,f"{SC}/r_oracle_score_goods.log")
    sh([PY,f"{SC}/build_oracle_tmpl.py","goods"],None,f"{SC}/r_build_oracle_goods.log")
    log("oracle goods gen")
    sh([PY,f"{SC}/gen_generic.py",f"{SC}/oracle_goods_jobs.json",f"{SC}/oracle_out"],PROT,f"{SC}/r_gen_oracle_goods.log")
    wait_seeds(f"{SC}/oracle_out",[f"{p}_oracle" for p in GOODS],timeout=21600)
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/oracle_out","goods","oracle",f"{SC}/oracle_goods_scores.json"],None,f"{SC}/r_score_oracle_goods.log")
    return summ(f"{SC}/oracle_goods_scores.json")
stage("oracle_goods",r4)

# R5: relaunch ESM
def r5():
    for g in [0,1,2,3]: subprocess.run(["bash",f"{ESM_SC}/esm1024_launch.sh",str(g)])
    time.sleep(20)
    n=int(subprocess.run("ps -eo args|grep -c '[r]un_foldbench_abag.py'",shell=True,capture_output=True,text=True).stdout.strip() or 0)
    return {"esm_workers":n}
stage("esm_relaunch",r5)
status["finished"]=time.strftime("%H:%M:%S"); save()
if os.path.exists(SENTINEL): os.remove(SENTINEL)
log("RESUME COMPLETE")
