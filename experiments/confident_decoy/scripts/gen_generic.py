import json,os,subprocess,threading,queue,sys
SC="/tmp/claude-1003/-home-ebrahim/c8e90e24-d694-4d7c-9a01-363182c820e2/scratchpad"
jobs=json.load(open(sys.argv[1]))          # jobs file
OUT=sys.argv[2]; os.makedirs(OUT,exist_ok=True)
q=queue.Queue()
for j in jobs: q.put(j)
done=[0]; lock=threading.Lock(); total=len(jobs)
def worker(gpu):
    while True:
        try: j=q.get_nowait()
        except queue.Empty: return
        seedcsv=",".join(str(s) for s in j["seeds"])
        env=dict(os.environ); env.update(CUDA_VISIBLE_DEVICES=str(gpu),LAYERNORM_TYPE="openfold",
            OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
        cmd=["/home/ebrahim/Protenix/.venv/bin/python","runner/inference.py","--model_name","protenix-v2",
             "--load_checkpoint_dir","/home/ebrahim/checkpoint","--input_json_path",j["json"],"--dump_dir",OUT,
             "--seeds",seedcsv,"--sample_diffusion.N_sample","5","--sample_diffusion.N_step","200","--use_template","true"]
        lf=f"{os.path.dirname(sys.argv[2])}/gen_{os.path.basename(sys.argv[2])}_{j['pdb']}.log"
        with open(lf,"w") as f:
            subprocess.run(cmd,cwd="/home/ebrahim/Protenix",env=env,stdout=f,stderr=subprocess.STDOUT)
        with lock: done[0]+=1; print(f"[{done[0]}/{total}] GPU{gpu} {j['pdb']} done",flush=True)
        q.task_done()
ts=[threading.Thread(target=worker,args=(g,)) for g in range(4)]
for t in ts: t.start()
for t in ts: t.join()
print("ALL_GEN_DONE",flush=True)
