import subprocess,time,os,json,glob
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad/selftmpl"
ESM_SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad"
PY="/home/ebrahim/Protenix/.venv/bin/python"
SENTINEL=f"{SC}/EXPERIMENTS_RUNNING"; STATUS=f"{SC}/overnight_status.json"
open(SENTINEL,"w").write("running")
status={"started":time.strftime("%Y-%m-%d %H:%M:%S"),"stages":{}}
def save(): json.dump(status,open(STATUS,"w"),indent=1)
def log(m): print(f"{time.strftime('%H:%M:%S')} {m}",flush=True); save()
def sh(cmd,cwd=None,logf=None):
    with open(logf,"w") as f: return subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT).returncode
def wait_seeds(out,names,n=50,timeout=18000,poll=60):
    t0=time.time()
    while time.time()-t0<timeout:
        cnt={nm:len(glob.glob(f"{out}/{nm}/seed_*")) for nm in names}
        if all(v>=n for v in cnt.values()): return True
        time.sleep(poll)
    return False
def read_summary(js):
    try: return json.load(open(js.replace(".json","_summary.json")))
    except: return None
DECOYS=list(json.load(open(f"{SC}/agonly_antigen_idx.json")))
GOODS=[]
for x in json.load(open(f"{SC}/safety15.json")):
    if x["pdb"] not in GOODS: GOODS.append(x["pdb"])

def stage(name,fn):
    status["stages"][name]={"status":"running","t":time.strftime("%H:%M:%S")}; save()
    try:
        r=fn(); status["stages"][name]={"status":"done","result":r,"t":time.strftime("%H:%M:%S")}
    except Exception as e:
        status["stages"][name]={"status":"FAILED","error":repr(e),"t":time.strftime("%H:%M:%S")}
    save()
    subprocess.run(["python3",f"{SC}/write_results.py"])

# ---- STAGE 1: score abonly decoys (gen already running externally) ----
def s1():
    log("wait abonly_decoys gen (12 pdbs x50)")
    ok=wait_seeds(f"{SC}/abonly_out",[f"{p}_abonly" for p in DECOYS])
    log(f"gen complete={ok}; scoring")
    sh([ "python3",f"{SC}/score_generic.py",f"{SC}/abonly_out","decoys","abonly",f"{SC}/abonly_scores.json"],logf=f"{SC}/ov_score_abonly_decoys.log")
    return read_summary(f"{SC}/abonly_scores.json")
stage("abonly_decoys",s1)

# ---- STAGE 2: oracle decoys build->gen->score ----
def s2():
    log("wait oracle_folds.json (decoy oracle lddt scoring)")
    t0=time.time()
    while not os.path.exists(f"{SC}/oracle_folds.json") and time.time()-t0<7200: time.sleep(30)
    sh(["python3",f"{SC}/build_oracle_tmpl.py","decoys"],logf=f"{SC}/ov_build_oracle_decoys.log")
    log("oracle decoy gen")
    sh([PY,f"{SC}/gen_generic.py",f"{SC}/oracle_decoys_jobs.json",f"{SC}/oracle_out"],cwd="/home/ebrahim/Protenix",logf=f"{SC}/ov_gen_oracle_decoys.log")
    wait_seeds(f"{SC}/oracle_out",[f"{p}_oracle" for p in DECOYS])
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/oracle_out","decoys","oracle",f"{SC}/oracle_scores.json"],logf=f"{SC}/ov_score_oracle_decoys.log")
    return read_summary(f"{SC}/oracle_scores.json")
stage("oracle_decoys",s2)

# ---- STAGE 3: abonly goods gen->score ----
def s3():
    log("abonly goods gen")
    sh([PY,f"{SC}/gen_generic.py",f"{SC}/abonly_goods_jobs.json",f"{SC}/abonly_out"],cwd="/home/ebrahim/Protenix",logf=f"{SC}/ov_gen_abonly_goods.log")
    wait_seeds(f"{SC}/abonly_out",[f"{p}_abonly" for p in GOODS])
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/abonly_out","goods","abonly",f"{SC}/abonly_goods_scores.json"],logf=f"{SC}/ov_score_abonly_goods.log")
    return read_summary(f"{SC}/abonly_goods_scores.json")
stage("abonly_goods",s3)

# ---- STAGE 4: oracle goods prep->score->build->gen->score ----
def s4():
    log("prep goods oracle (chainmap + native chains)")
    sh([PY,f"{SC}/prep_oracle_goods.py"],logf=f"{SC}/ov_prep_oracle_goods.log")
    log("goods oracle lddt scoring")
    sh([PY,f"{SC}/build_oracle_score.py",f"{SC}/oracle_chainmap_goods.json",f"{SC}/native_chains_goods",f"{SC}/oracle_folds_goods.json"],logf=f"{SC}/ov_oracle_score_goods.log")
    sh(["python3",f"{SC}/build_oracle_tmpl.py","goods"],logf=f"{SC}/ov_build_oracle_goods.log")
    log("oracle goods gen")
    sh([PY,f"{SC}/gen_generic.py",f"{SC}/oracle_goods_jobs.json",f"{SC}/oracle_out"],cwd="/home/ebrahim/Protenix",logf=f"{SC}/ov_gen_oracle_goods.log")
    wait_seeds(f"{SC}/oracle_out",[f"{p}_oracle" for p in GOODS])
    sh(["python3",f"{SC}/score_generic.py",f"{SC}/oracle_out","goods","oracle",f"{SC}/oracle_goods_scores.json"],logf=f"{SC}/ov_score_oracle_goods.log")
    return read_summary(f"{SC}/oracle_goods_scores.json")
stage("oracle_goods",s4)

# ---- STAGE 5: relaunch ESM on GPUs 0-3 ----
def s5():
    for g in [0,1,2,3]:
        subprocess.run(["bash",f"{ESM_SC}/esm1024_launch.sh",str(g)])
    time.sleep(20)
    n=int(subprocess.run("ps -eo args|grep -c '[r]un_foldbench_abag.py'",shell=True,capture_output=True,text=True).stdout.strip() or 0)
    return {"esm_workers":n}
stage("esm_relaunch",s5)

status["finished"]=time.strftime("%Y-%m-%d %H:%M:%S"); save()
os.remove(SENTINEL)
log("ALL OVERNIGHT STAGES COMPLETE")
