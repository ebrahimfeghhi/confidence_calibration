#!/usr/bin/env bash
# Validate the single-load driver on a WELL-BEHAVED interface, after the 9s36 smoke test.
#
# Exact reproduction of the shell version is NOT the test: seven stages in one process is a different
# RNG history than seven fresh processes, and we know history survives seed_everything (the 50-seeds-
# in-one-process sweep does not reproduce standalone). So instead we check that a target the model
# reliably gets right STAYS right. 9h1f_AB solves on 50/50 seeds; the shell version gave donors
# 0.881-0.900 at ipTM ~0.934 and arms 0.895-0.906 at 0.934-0.937. Landing in that band means the
# driver is sound; DockQ collapsing or the arms scattering means it is not.
set -u
REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
cd "$REPO" || exit 1
while pgrep -u "$(id -u)" -f commit11_fast.py >/dev/null; do sleep 20; done
echo "=== 9s36 smoke finished, starting 9h1f validation $(date +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=5 LAYERNORM_TYPE=openfold \
  /home/ebrahim/Protenix/.venv/bin/python -u scripts/commit11_fast.py \
  --pdb 9h1f --iface 9h1f_AB --seeds 300000 --out artifacts/c11fast 2>&1 \
  | tee ops/c11fast_9h1f.log
