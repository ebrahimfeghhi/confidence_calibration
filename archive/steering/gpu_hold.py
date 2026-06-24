"""Reserve a GPU: hold a chunk of memory + a light perpetual matmul so the GPU shows BUSY
and no one else grabs it. Writes NOTHING to disk (no quota risk). CUDA_VISIBLE_DEVICES picks
the GPU; kill the process (SIGTERM / Ctrl-C) to release it instantly.

Usage:  CUDA_VISIBLE_DEVICES=N /home/ebrahim/esm/.venv/bin/python gpu_hold.py --gb 75
"""
import sys
import time

import torch


def _flag(name, default):
    if name in sys.argv:
        return float(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    gb = _flag("--gb", 75.0)
    dev = "cuda:0"                       # CUDA_VISIBLE_DEVICES remaps the chosen GPU to cuda:0
    chunks, got_mb = [], 0
    while got_mb < gb * 1024:            # reserve memory in 256 MB chunks (robust vs one huge alloc)
        try:
            chunks.append(torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev))  # 256 MB
            got_mb += 256
        except RuntimeError:
            break
    print(f"[gpu_hold] reserved ~{got_mb/1024:.1f} GB on {dev}; holding. Kill to release.", flush=True)
    a = torch.randn(8192, 8192, device=dev)
    while True:                          # light perpetual compute -> nonzero util, bounded, no growth
        a = torch.sigmoid(a @ a)
        torch.cuda.synchronize()
        time.sleep(1.0)


if __name__ == "__main__":
    main()
