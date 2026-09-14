#!/usr/bin/env python
# guard_gpu.py <GB> <stop_file>
# Reserves <GB> of VRAM on the CUDA_VISIBLE_DEVICES card by holding a dummy tensor, so another
# user's process sees little free memory and their scheduler avoids this card. Releases on stop_file.
import torch, time, os, sys
gb = float(sys.argv[1]); stop = sys.argv[2]
n = int(gb * (1024 ** 3) / 2)                      # float16 = 2 bytes/elt
x = torch.empty(n, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()
print(f"[guard] reserved {gb:.0f} GB on visible GPU; holding until {stop}", flush=True)
while not os.path.exists(stop):
    time.sleep(10)
del x; torch.cuda.empty_cache()
print("[guard] released", flush=True)
