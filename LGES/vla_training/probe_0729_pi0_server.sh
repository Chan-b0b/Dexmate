#!/usr/bin/env bash
# π0 offline probe battery — SERVER version (local checkpoints, no HF).
# Three arms of the 0729 π0 round (run_pi0_film_frombase_0729.sh / run_pi0_naive_0729.sh):
#   naive  = pi0_naive_0729                  (no FiLM — raw-template baseline)
#   state  = pi0_film_frombase_state_0729    (FiLM on the STATE token — SmolVLA-'prefix' analogue)
#   action = pi0_film_frombase_action_0729   (FiLM on the ACTION tokens — 'suffix' analogue)
#
# HOW TO READ THIS (scope limit — do not skip):
#   π0 answers the TOKEN-level injection question only (state vs action, both in
#   embed_suffix); it is NOT literally SmolVLA's prefix-vs-suffix, and it keeps the OLD
#   calibration (3 channels contact,fz,seal, FZ_OFF=2.1) — so compare arms against each
#   other and against the pi05 cells for sign/shape, never magnitudes across architectures.
#
# The FiLM env below MUST match training (run_pi0_film_frombase_0729.sh): cond=contact,fz,
# seal, mask_force=1, FZ_OFF=2.1, F0/TAU/FZ_TAU=6/4/5 (film_contact defaults). Unlike the
# pi05 battery, F0/TAU/FZ_TAU are exported EXPLICITLY so eval_offline's different env
# defaults (12/10/30) can't silently re-calibrate the eval cells.
#
# Requires the pi0 probe routing added 2026-08-12: probe_state_authority/probe_press_sim/
# eval_offline take --film-pi0, which patches via film_contact_pi0 (MEAN_STD state) AND
# installs the forced-c hook on film_contact_pi0._condition_from_state.
#
# Usage: ./probe_0729_pi0_server.sh                    (GPU=<n>, default 7)
#        GPU=5 RUNS=state ./probe_0729_pi0_server.sh   # one arm per GPU, in parallel
#        GPU=4 RUNS=naive CKPTS=best ./probe_0729_pi0_server.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
RV="$DIR/datasets/lges_case_pick_0729_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val

# training-matched FiLM calibration (see header) — explicit incl. F0/TAU/FZ_TAU
export FILM_COND=contact,fz,seal
export FILM_MASK_FORCE=1
export FILM_FZ_OFF=2.1
export FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5

[[ -d "$RV/meta" ]] || { echo "[pi0probe] val dataset missing: $RV" >&2; exit 1; }
mkdir -p "$DIR/probes"

for arm in ${RUNS:-naive state action}; do
  case "$arm" in
    naive)  RUN="$DIR/outputs/pi0_naive_0729";               FLAGS=(--naive);     TAG="pi0naive" ;;
    state)  RUN="$DIR/outputs/pi0_film_frombase_state_0729"; FLAGS=(--film-pi0);  TAG="pi0filmstate";  export FILM_INJECT=state ;;
    action) RUN="$DIR/outputs/pi0_film_frombase_action_0729"; FLAGS=(--film-pi0); TAG="pi0filmaction"; export FILM_INJECT=action ;;
    *)  echo "[pi0probe] unknown arm '$arm' (want naive|state|action)" >&2; exit 1 ;;
  esac
  [[ -d "$RUN/checkpoints" ]] || { echo "[pi0probe] skip $arm (no checkpoints)"; continue; }

  for ck in ${CKPTS:-best last}; do
    CK="$RUN/checkpoints/$ck"
    [[ -e "$CK" ]] || { echo "[pi0probe] skip $arm/$ck (missing)"; continue; }
    state() { # <outfile> <extra...>
      local out=$1; shift
      "$PY" "$DIR/probe_state_authority.py" "${FLAGS[@]}" --all-episodes \
        --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" "$@" \
        > "$DIR/probes/$out" 2>&1
      echo "[pi0probe] $out rc=$? -> $(grep -a -m1 'ALL frames' "$DIR/probes/$out" | sed 's/^ *//')"
    }
    state "0729_state_${TAG}_${ck}_pc_fc.txt"  --swap firstcontact --pre-contact 10
    state "0729_state_${TAG}_${ck}_ramp8.txt"  --swap fcscale --fc-mag 8
    state "0729_state_${TAG}_${ck}_ramp12.txt" --swap fcscale --fc-mag 12
    state "0729_state_${TAG}_${ck}_pc_r12.txt" --swap fcscale --fc-mag 12 --pre-contact 10
    for off in 1 30; do
      "$PY" "$DIR/probe_press_sim.py" "${FLAGS[@]}" --checkpoint "$CK" \
        --dataset-root "$RV" --repo-id "$REPO_VAL" \
        --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
        > "$DIR/probes/0729_sim_${TAG}_${ck}_off${off}_fzdelta.txt" 2>&1
      echo "[pi0probe] sim $arm/$ck off$off rc=$?"
    done
    EVAL_FLAGS=("${FLAGS[@]}")
    [[ "$arm" == naive ]] && EVAL_FLAGS=()   # eval_offline has no --naive; plain load
    "$PY" "$DIR/eval_offline.py" "${EVAL_FLAGS[@]}" --checkpoint "$CK" \
      --val-root "$RV" --repo-id "$REPO_VAL" \
      > "$DIR/probes/0729_eval_${TAG}_${ck}.txt" 2>&1
    echo "[pi0probe] eval $arm/$ck rc=$?"
  done
done

echo; echo "===== pi0 verdict material (token-level injection question) ====="
grep -a -H "ALL frames" "$DIR"/probes/0729_state_pi0*_*.txt 2>/dev/null
grep -a -H -A3 "── summary" "$DIR"/probes/0729_sim_pi0*_*.txt 2>/dev/null | grep -v "^--$"
grep -a -H "OVERALL" "$DIR"/probes/0729_eval_pi0*_*.txt 2>/dev/null
echo "[pi0probe] DONE — commit probes/ + push"
