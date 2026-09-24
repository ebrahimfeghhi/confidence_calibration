#!/usr/bin/env python
"""Occupy GPUs so they are not taken while a sweep is paused.

This is a shared machine and an idle card gets claimed within minutes. Allocating a small tensor is
enough to show up in nvidia-smi as in use.

    CUDA_VISIBLE_DEVICES=1 hold_gpus.py --gpus 1            hold until killed
    CUDA_VISIBLE_DEVICES=1 hold_gpus.py --gpus 1 --minutes 30   hold, then release

Kill the tmux session (or the process) to release. Holders must be stopped before relaunching real
work on the same card -- the sweep needs that memory.
"""
import argparse
import time

import torch


def hold(gb=1.0):
    """in: gigabytes to occupy.  out: the tensor, kept alive by the caller."""
    n = int(gb * (1024 ** 3) / 4)
    return torch.zeros(n, dtype=torch.float32, device="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True, help="comma-separated device ids")
    ap.add_argument("--gb", type=float, default=1.0, help="per GPU")
    ap.add_argument("--minutes", type=float, default=0, help="0 = until killed")
    A = ap.parse_args()

    # ONE PROCESS PER GPU, launched with CUDA_VISIBLE_DEVICES set, so each holder is killable on its
    # own and the device is always index 0 here. A single process calling set_device(g) fails outright
    # when the caller's shell exports an empty CUDA_VISIBLE_DEVICES, which is how this first broke.
    held = [hold(A.gb)]
    print(f"  holding {A.gpus} with {A.gb} GB (device 0 of this process)", flush=True)
    deadline = time.time() + A.minutes * 60 if A.minutes else None
    while deadline is None or time.time() < deadline:
        time.sleep(30)
    print("  released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
