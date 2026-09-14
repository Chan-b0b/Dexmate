#!/usr/bin/env bash
# ACT offline probe battery — SERVER version (local checkpoints, no HF).
# Arms of the 0729 ACT round (run_act_0729.sh):
#   naive = act_0729               (no FiLM — raw-template baseline, from scratch)
#   state = act_film_scratch_0729  (encoder state-token FiLM, scratch — 'frombase state')
# (act_film_layers_0729 from the inject='layers' round can be probed with
#  RUNS=layers once its training finishes — FILM_INJECT=layers is set per-arm below.)
#
# Same cells and training-matched calibration as probe_0729_pi0_server.sh (cond=contact,
# fz,seal mask_force=1 FZ_OFF=2.1 F0/TAU/FZ_TAU=6/4/5, exported explicitly so
# eval_offline's 12/10/30 env defaults can't silently re-calibrate the eval cells).
#
# Requires the ACT probe routing (--film-act) + forced-c hook on
# film_contact_act._condition_from_state (2026-08-14).
#
# Usage: ./probe_0729_act_server.sh                   (GPU=<n>, default 7)
#        GPU=4 RUNS=naive ./probe_0729_act_server.sh  # one arm per GPU, in parallel
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
RV="$DIR/datasets/lges_case_pick_0729_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val

export FILM_COND=contact,fz,seal
export FILM_MASK_FORCE=1
export FILM_FZ_OFF=2.1
export FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5

[[ -d "$RV/meta" ]] || { echo "[actprobe] val dataset missing: $RV" >&2; exit 1; }
mkdir -p "$DIR/probes"

for arm in ${RUNS:-naive state}; do
  case "$arm" in
    naive)  RUN="$DIR/outputs/act_0729";              FLAGS=(--naive);    TAG="actnaive" ;;
    state)  RUN="$DIR/outputs/act_film_scratch_0729"; FLAGS=(--film-act); TAG="actfilmstate";  export FILM_INJECT=state ;;
    layers) RUN="$DIR/outputs/act_film_layers_0729";  FLAGS=(--film-act); TAG="actfilmlayers"; export FILM_INJECT=layers ;;
    *)  echo "[actprobe] unknown arm '$arm' (want naive|state|layers)" >&2; exit 1 ;;
  esac
  [[ -d "$RUN/checkpoints" ]] || { echo "[actprobe] skip $arm (no checkpoints)"; continue; }

  for ck in ${CKPTS:-best last}; do
    CK="$RUN/checkpoints/$ck"
    [[ -e "$CK" ]] || { echo "[actprobe] skip $arm/$ck (missing)"; continue; }
    state() { # <outfile> <extra...>
      local out=$1; shift
      "$PY" "$DIR/probe_state_authority.py" "${FLAGS[@]}" --all-episodes \
        --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" "$@" \
        > "$DIR/probes/$out" 2>&1
      echo "[actprobe] $out rc=$? -> $(grep -a -m1 'ALL frames' "$DIR/probes/$out" | sed 's/^ *//')"
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
      echo "[actprobe] sim $arm/$ck off$off rc=$?"
    done
    EVAL_FLAGS=("${FLAGS[@]}")
    [[ "$arm" == naive ]] && EVAL_FLAGS=()   # eval_offline has no --naive; plain load
    "$PY" "$DIR/eval_offline.py" "${EVAL_FLAGS[@]}" --checkpoint "$CK" \
      --val-root "$RV" --repo-id "$REPO_VAL" \
      > "$DIR/probes/0729_eval_${TAG}_${ck}.txt" 2>&1
    echo "[actprobe] eval $arm/$ck rc=$?"
  done
done

echo; echo "===== ACT verdict material ====="
grep -a -H "ALL frames" "$DIR"/probes/0729_state_act*_*.txt 2>/dev/null
grep -a -H -A3 "── summary" "$DIR"/probes/0729_sim_act*_*.txt 2>/dev/null | grep -v "^--$"
grep -a -H "OVERALL" "$DIR"/probes/0729_eval_act*_*.txt 2>/dev/null
echo "[actprobe] DONE — commit probes/ + push"
