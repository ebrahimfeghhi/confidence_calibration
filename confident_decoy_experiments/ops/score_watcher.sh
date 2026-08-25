#!/bin/bash
# Auto-score allpose arms as soon as their generation finishes.
#
# Polls every 5 min. For each configured arm: if all 2500 seeds have their final sample JSON and no
# <run>_scores.json exists yet, copy the arm into the repo (project rule: everything under tts_abag),
# verify template/MSA conditions, then score it.
#
# SERIALISED ON PURPOSE: one scoring job at a time. OST ignores taskset (it rebinds itself), so the
# only real lever on CPU is how many run at once -- two arms in parallel would blow the 64-core target
# while the GPU sweep is also running.
#
# CONFIG IS EXPLICIT, NOT INFERRED. Each arm's interface chains and donor cache were validated by
# checking the cache's max DockQ against free_best_dockq in native_arm_scores.json:
#   8fas_BC 0.987 | 9jbq_AC 0.584 | 8av2_AC 0.612 | 8slb_AB 0.462 | 8slb_AC 0.655
# Getting --donor-cache wrong does NOT error -- donor_dockq silently becomes NaN.

set -uo pipefail

REPO=/home/ebrahim/tts_abag/confident_decoy_experiments
OLD=/home/ebrahim/confident_decoy_experiments/artifacts
NEW=$REPO/artifacts
PY=/home/ebrahim/Protenix/.venv/bin/python
LOG=$REPO/ops/score_watcher.log
INTERVAL=300

# run|pdb|ifaces|extra|type
#   type=allpose  -> score_allpose.py,  extra = --donor-cache (free-pool OST dir for donor_dockq)
#   type=free     -> score_freedeep.py, extra = optional --prior file of scored seeds 0-150
# Expected seed count is read from each arm's own jobs.json, NOT hardcoded: allpose arms declare
# 2500 seeds, the free arms 2349 (151-2499, extending the existing 0-150 to 2500 total).
ARMS=(
  "allpose8av210|8av2|8av2_AC:A:C|decoy_set_ost|allpose"
  "allpose8fas10nomsa|8fas|8fas_BC:B:C|decoy_set_ost|allpose"
  # allpose8slb10 is COMPLETE and SCORED (8slb_AB 0.021, 8slb_AC 0.009 -- both genuine failures).
  # Deliberately NOT listed: its only remaining effect here would be maybe_queue_nomsa re-appending
  # allpose8slb10nomsa, which was cancelled by request. Re-add the line if you want that arm back.
  #   "allpose8slb10|8slb|8slb_AB:A:B,8slb_AC:A:C|decoy_set_ost|allpose"
  "freedeep8fas|8fas|8fas_BC:B:C||free"
  "freedeep9jbq|9jbq|9jbq_AC:A:C||free"
  "freedeep8av2|8av2|8av2_AC:A:C||free"
  "freedeep8slb|8slb|8slb_AB:A:B,8slb_AC:A:C||free"
  "freedeep8pn0|8pn0|8pn0_BE:B:E,8pn0_AE:A:E||free"      # AE is free data: same OST call
  "freedeep8pn0low|8pn0|8pn0_BE:B:E,8pn0_AE:A:E||free"   # seeds 0-100, scored separately
  "freedeep8q3j|8q3j|8q3j_AB:A:B,8q3j_AC:A:C||free"
  "freedeep8aok|8aok|8aok_AB:A:B||free"
  # Templated twins of the three new complexes. donor-cache validated against free_best_dockq:
  #   8pn0_BE extra_set_ost 0.741 OK | 8q3j_AC decoy_set_ost 0.531 OK | 8aok_AB decoy_set_ost 0.695 OK
  # NOTE 8q3j_AB has NO cached free poses in any cache, so its donor_dockq will be NaN (selection
  # is unaffected -- donor_dockq is descriptive only).
  "allpose8pn010|8pn0|8pn0_BE:B:E,8pn0_AE:A:E|extra_set_ost|allpose"
  "allpose8q3j10|8q3j|8q3j_AB:A:B,8q3j_AC:A:C|decoy_set_ost|allpose"
  "allpose8aok10|8aok|8aok_AB:A:B|decoy_set_ost|allpose"
  "allpose8aok10nomsa|8aok|8aok_AB:A:B|decoy_set_ost|allpose"
  "allpose8slb10nomsa|8slb|8slb_AB:A:B,8slb_AC:A:C|decoy_set_ost|allpose"
  "allpose9jbq10nomsa|9jbq|9jbq_AC:A:C|extra_set_ost|allpose"
  "allpose8av210nomsa|8av2|8av2_AC:A:C|decoy_set_ost|allpose"
  "allpose8che10nomsa|8che|8che_AE:A:E,8che_BE:B:E|decoy_set_ost|allpose"
  # TRAIN-split arms at 250x5. donor-cache = abag_all_ost: these interfaces have no
  # <iface>_<seed>_<sample> cache, only <pdb>_<seed>_<sample>, so donor_dockq will be NaN.
  # That field is descriptive only -- selection and DockQ are unaffected.
  "allpose8dtn5|8dtn|8dtn_AB:A:B|abag_all_ost|allpose"
  "allpose8z8v5|8z8v|8z8v_AB:A:B|abag_all_ost|allpose"
  "allpose8a445|8a44|8a44_AC:A:C,8a44_AD:A:D|abag_all_ost|allpose"
  "allpose8vzo5|8vzo|8vzo_AB:A:B,8vzo_AC:A:C|abag_all_ost|allpose"
  "allpose8ivx5|8ivx|8ivx_AB:A:B,8ivx_AC:A:C|abag_all_ost|allpose"
  "allpose8oxw5|8oxw|8oxw_AB:A:B,8oxw_AC:A:C|abag_all_ost|allpose"
  "allpose8sfz5|8sfz|8sfz_AD:A:D|abag_all_ost|allpose"
  "allpose8the5|8the|8the_AB:A:B|abag_all_ost|allpose"
  "allpose8udz5|8udz|8udz_AC:A:C,8udz_AD:A:D|abag_all_ost|allpose"
  "allpose7zau5|7zau|7zau_AB:A:B|abag_all_ost|allpose"
  # UNTEMPLATED 1024-seed arms extending each train complex's free pool. type=free so the
  # allpose "Found 0 templates" gate does NOT apply -- these use the base FoldBench json,
  # whose natural hmmsearch search legitimately returns none for some chains.
  "free1k8dtn|8dtn|8dtn_AB:A:B||free"
  "free1k8z8v|8z8v|8z8v_AB:A:B||free"
  "free1k8a44|8a44|8a44_AC:A:C,8a44_AD:A:D||free"
  "free1k8vzo|8vzo|8vzo_AB:A:B,8vzo_AC:A:C||free"
  "free1k8ivx|8ivx|8ivx_AB:A:B,8ivx_AC:A:C||free"
  "free1k8oxw|8oxw|8oxw_AB:A:B,8oxw_AC:A:C||free"
  "free1k8sfz|8sfz|8sfz_AD:A:D||free"
  "free1k8the|8the|8the_AB:A:B||free"
  "free1k8udz|8udz|8udz_AC:A:C,8udz_AD:A:D||free"
  "free1k7zau|7zau|7zau_AB:A:B||free"
)

QUEUE=$REPO/ops/run_queue.txt
THRESH=0.23
# Auto-queueing a no-MSA twin is OFF for the 250x5 train round. Every train arm still gets SCORED and
# its verdict logged; only the automatic follow-up run is suppressed. Reason: the twins are built at
# 250x10, and most train arms are confident decoys that will trip THRESH, so leaving this on would
# silently append ~9 arms x 2500 seeds (~3x the approved compute) to the GPU queue.
# Re-enable for a single watcher run with:  AUTO_NOMSA=1 bash ops/score_watcher.sh
AUTO_NOMSA=${AUTO_NOMSA:-0}

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# After an arm scores: if pooled ipTM selection failed on ANY of its interfaces, build the no-MSA
# twin and append it to the GPU queue. Rationale: on 8fas_BC the MSA-on sweep selected 0.174 against
# a 0.992 oracle while the no-MSA arm reached 0.982 -- when selection is the failure, stripping the
# MSA is the contrast that has actually moved it. Skips arms that are already no-MSA.
maybe_queue_nomsa() {
    local RUN=$1
    case "$RUN" in *nomsa) return 0;; esac
    # Gate is here, not at the call site, so the verdict below is still computed and logged.
    [ -f "$NEW/${RUN}_scores.json" ] || return 0

    local verdict
    verdict=$($PY - "$NEW/${RUN}_scores.json" "$THRESH" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); thresh = float(sys.argv[2])
rows = d["rows"]
labs = sorted({k for r in rows for k in r if k.endswith(tuple("ABCDEFGH")) and isinstance(r[k], dict)})
fails, parts = [], []
for lab in labs:
    v = [r[lab] for r in rows if lab in r]
    if not v:
        continue
    sel = max(v, key=lambda x: x["sel_iptm"])["out_sel"]
    orc = max(x["out_max"] for x in v)
    parts.append(f"{lab} sel={sel:.3f} oracle={orc:.3f}")
    if sel < thresh:
        fails.append(lab)
print(("FAIL " if fails else "PASS ") + "; ".join(parts) + (" | failed: " + ",".join(fails) if fails else ""))
PYEOF
)
    say "$RUN verdict: $verdict"
    case "$verdict" in
        FAIL*) ;;
        *) return 0;;
    esac
    if [ "$AUTO_NOMSA" != "1" ]; then
        say "$RUN: selection failed, but AUTO_NOMSA=0 -- NOT queueing ${RUN}nomsa (250x5 train round)"
        return 0
    fi

    if grep -qx "${RUN}nomsa" "$QUEUE"; then
        say "$RUN: ${RUN}nomsa already queued"; return 0
    fi

    say "$RUN: FAILED selection -- building no-MSA twin ${RUN}nomsa"
    if ! CDE_ARTIFACTS_DIR=$OLD $PY "$REPO/scripts/build_nomsa_variant.py" --run "$RUN" >>"$LOG" 2>&1; then
        say "*** $RUN: build_nomsa_variant FAILED -- not queued, see $LOG ***"; return 1
    fi
    if ! CDE_ARTIFACTS_DIR=$OLD $PY "$REPO/scripts/run_health.py" --preflight \
           --jobs "$OLD/${RUN}nomsa_jobs.json" --out "$OLD/sweep_out/${RUN}nomsa" \
           --logs "$OLD/${RUN}nomsa_logs" --seeds 10 >>"$LOG" 2>&1; then
        say "*** $RUN: preflight FAILED for ${RUN}nomsa -- not queued ***"; return 1
    fi
    echo "${RUN}nomsa" >> "$QUEUE"
    say "$RUN: queued ${RUN}nomsa (built + preflighted). GPU watcher will start it."
}

seeds_done() {  # $1 = run; count in whichever root has more
    local a b
    a=$(find "$OLD/sweep_out/$1" -name '*_summary_confidence_sample_4.json' 2>/dev/null | wc -l)
    b=$(find "$NEW/sweep_out/$1" -name '*_summary_confidence_sample_4.json' 2>/dev/null | wc -l)
    [ "$a" -gt "$b" ] && echo "$a" || echo "$b"
}

# Match the INTERPRETER RUNNING THE SCRIPT, not any command line containing the string. The bare
# pattern also matched `tee ops/score_allpose8a445.log`, so a finished scoring job left a lingering
# tee/pane that looked like a running scorer forever -- the watcher then refused to score anything
# else and sat printing "another scoring job is running" for hours while arms piled up unscored.
scoring_running() { pgrep -u "$(id -un)" -f "python.*scripts/score_allpose\.py" >/dev/null 2>&1; }

say "score watcher started (every $((INTERVAL/60)) min, one arm at a time)"

while true; do
    all_done=1
    for spec in "${ARMS[@]}"; do
        IFS='|' read -r RUN PDB IFACES EXTRA TYPE <<< "$spec"
        NSEEDS=$($PY -c "import json;print(sum(len(j['seeds']) for j in json.load(open('$OLD/${RUN}_jobs.json'))))" 2>/dev/null)
        if [ -z "$NSEEDS" ]; then
            say "$RUN: no jobs.json yet -- not built, skipping"; continue
        fi

        if [ -f "$NEW/${RUN}_scores.json" ]; then
            # Only allpose arms can have a no-MSA twin -- a free arm has no injected template, and
            # its scores.json has a different shape the verdict parser would misread.
            [ "$TYPE" = "allpose" ] && maybe_queue_nomsa "$RUN"   # idempotent; no-ops once queued
            continue
        fi
        all_done=0

        n=$(seeds_done "$RUN")
        if [ "$n" -lt "$NSEEDS" ]; then
            say "$RUN: $n/$NSEEDS seeds -- still generating"
            continue
        fi

        if scoring_running; then
            say "$RUN: ready, but another scoring job is running -- waiting"
            continue
        fi

        # --- copy into the repo (rule: everything under tts_abag) -------------
        say "$RUN: generation COMPLETE ($n seeds). Copying into repo ..."
        mkdir -p "$NEW/sweep_out"
        rsync -a "$OLD/sweep_out/$RUN" "$NEW/sweep_out/" 2>/dev/null
        rsync -a "$OLD/${RUN}_jobs.json" "$OLD/${RUN}_donors.json" \
                 "$OLD/${RUN}_status.json" "$NEW/" 2>/dev/null
        rsync -a "$OLD/${RUN}_logs" "$NEW/" 2>/dev/null

        # --- verify conditions BEFORE scoring --------------------------------
        nulls=$(grep -h "Found 0 templates" "$NEW/${RUN}_logs"/*.log 2>/dev/null | wc -l)
        loaded=$(grep -h "Found 1 templates" "$NEW/${RUN}_logs"/*.log 2>/dev/null | wc -l)
        msa=$(grep -ho "N_msa [0-9]*" "$NEW/${RUN}_logs"/*.log 2>/dev/null | sort -u | tr '\n' ' ')

        if [ "$TYPE" = "allpose" ]; then
            # Injected single-donor templates: every chain load must report exactly one.
            if [ "$nulls" -gt 0 ]; then
                say "*** $RUN: REFUSING TO SCORE -- 'Found 0 templates' x$nulls (untemplated arm) ***"
                continue
            fi
            if [ "$loaded" -eq 0 ]; then
                say "*** $RUN: REFUSING TO SCORE -- no 'Found 1 templates' lines; loading UNVERIFIED ***"
                continue
            fi
            case "$RUN" in
                *nomsa) [ "$msa" = "N_msa 1 " ] || say "*** $RUN: WARNING expected 'N_msa 1', got '$msa' ***" ;;
            esac
        else
            # FREE arms use the base FoldBench json: the NATURAL hmmsearch template search, which
            # legitimately returns many templates (or none for a chain with no hits). So
            # 'Found 0 templates' is NOT a failure here and must not gate scoring -- applying the
            # allpose check would refuse every free arm. The MSA must still be present.
            case "$msa" in
                *"N_msa 1 "*) say "*** $RUN: WARNING free arm reports N_msa 1 -- MSA missing? ***" ;;
            esac
            say "$RUN: free arm (natural templates; 'Found 0' not a failure here) | $msa"
        fi
        say "$RUN: conditions OK (templates $loaded loaded / $nulls null | $msa) -- scoring now"

        tmux kill-session -t score 2>/dev/null
        tmux new-session -d -s score -c "$REPO"
        if [ "$TYPE" = "allpose" ]; then
            tmux send-keys -t score \
              "$PY scripts/score_allpose.py --run $RUN --pdb $PDB --ifaces $IFACES \
               --donor-cache $EXTRA --workers 8 --cores 0-47 2>&1 | tee ops/score_${RUN}.log" Enter
        else
            PRIOR_ARG=""; [ -n "$EXTRA" ] && PRIOR_ARG="--prior $EXTRA"
            tmux send-keys -t score \
              "$PY scripts/score_freedeep.py --run $RUN --pdb $PDB --ifaces $IFACES \
               $PRIOR_ARG --workers 8 --cores 0-47 2>&1 | tee ops/score_${RUN}.log" Enter
        fi
        say "$RUN: launched in tmux 'score' ($TYPE)"
        break                                              # one at a time
    done

    if [ "$all_done" -eq 1 ] && ! scoring_running; then
        say "all configured arms scored -- watcher exiting"
        exit 0
    fi
    sleep "$INTERVAL"
done
