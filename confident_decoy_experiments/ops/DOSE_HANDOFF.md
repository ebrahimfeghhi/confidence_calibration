# Dose-response sweep — built, NOT launched

Blocked by the Claude Code safety classifier on 2026-08-21 ~20:00. Everything below is written,
syntax-checked, and ready. Three commands stand between it and running.

## What it is

40 interfaces drawn at random (seed 20260821) from the 172 with 50-seed reference data — a random
sample, unlike the 43 arms of the 1e-2 sweep, which were picked because MSA subsampling rescued them.
That selection is why the current numbers cannot say whether weight noise helps in general.

6 rungs x 10 seeds x 40 interfaces = 2400 seeds. σ = 0 (control), 1e-3, 5e-3, 1e-2, 5e-2, 1e-1;
scope msa+pair, per-cycle noise, dropout off, full MSA.

Estimated 6–20 h on 4 GPUs (median 34 s/seed, p90 140 s/seed over 101 completed arms).

Queue order is **interface-major**: all 6 rungs of one interface before moving on. If it is cut
short, what exists is complete curves on a subsample, not one rung measured everywhere.

## To launch

    cd /home/ebrahim/tts_abag/confident_decoy_experiments

1. Retire the old tiers (graceful — workers exit after their current arm, nothing is killed):

    rm -f artifacts/.e52_enabled artifacts/.e1_enabled
    mkdir -p artifacts/retired_wnoise_tiers
    mv artifacts/esc_*_jobs.json artifacts/e52_*_jobs.json artifacts/e1_*_jobs.json \
       artifacts/retired_wnoise_tiers/

   Then stop the escalation driver, which otherwise keeps manufacturing 1e-2 work:
   `tmux kill-session -t escdrv` (also: escgrab, escw1, escw5, escw6, escw7, escscore,
   eschealth, chain52, chain1e1).

2. Start the sweep:

    tmux new-session -d -s dosegrab   -c $PWD; tmux send-keys -t dosegrab   "bash ops/dose_grabber.sh 2>&1 | tee -a ops/dose_grabber.log" Enter
    tmux new-session -d -s dosewatch  -c $PWD; tmux send-keys -t dosewatch  "bash ops/dose_watch.sh 2>&1 | tee -a ops/dose_watch.log" Enter
    tmux new-session -d -s dosehealth -c $PWD; tmux send-keys -t dosehealth "bash ops/dose_health.sh 2>&1 | tee -a ops/dose_health.log" Enter

The grabber starts one worker per idle GPU on cards 1/5/6/7, capped at 4, and only takes a card that
is free of other users **and** under 2 GB used — so it will not double-book a card still finishing an
old 1e-2 arm.

## Files

| path | role |
|---|---|
| `artifacts/dose_queue.txt` | 240 arms, interface-major order |
| `artifacts/dose_sample.json` | the 40 drawn interfaces + draw seed |
| `ops/dose_worker.sh` | per-GPU worker, mkdir-atomic claiming |
| `ops/run_dose.sh` | runner; asserts σ=0 really applied no noise |
| `ops/dose_grabber.sh` | acquires idle GPUs, idles rather than exits |
| `ops/dose_watch.sh` | every 30 min: score, then extension check |
| `ops/dose_health.sh` | every 10 min: errors, load, disk |
| `scripts/score_dose.py` | per-arm scoring + the σ curve |
| `scripts/dose_extend.py` | ladder extension, then deepening |

## Automatic behaviour once running

**Ladder extension** (you approved this): if an end rung beats its inward neighbour by >0.010 mean
selected DockQ, the next rung outward is added for all 40 interfaces. Low: 5e-4, 1e-4, 5e-5.
High: 5e-1, 1.0, 5.0. Guarded by a minimum of 12 complete curves, so it cannot chase noise from the
handful of small fast interfaces that finish first.

**Deepening** (your last message): when the ladder stops widening and every arm is complete, all arms
get +10 seeds and re-run, repeating until you stop it.

## Reading the result

`scripts/score_dose.py` prints sel_mean, sel_median, oracle_mean, frac_acceptable per σ, over
interfaces with a complete curve. Watch the selected and oracle columns together — a flat selected
curve under a rising oracle is an ipTM selection failure, not an absence of effect.

## Caveat

10 seeds per point is thin for any single interface; one lucky basin hit moves it a lot. The mean over
40 should absorb that, but per-interface claims off this data would not be safe.
