#!/usr/bin/env bash
# pi05_naive_0729 offline probe battery — SERVER version (local checkpoints, no HF).
# Architecture-generality test: does the bypass/template finding hold beyond SmolVLA?
#   Prediction: pi05 naive = raw template responder — pc_fc responds (~100%),
#   ramp8->ramp12 flat-or-declining, press-sim (seal-never) over-penetrates.
# Prereq: git pull first — needs the policy-agnostic probe loading
#   (probe_state_authority/probe_press_sim/eval_offline load via factory + the
#   train_pi05 preprocessor shim; guarded import works on lerobot 0.5.1 AND newer).
# Usage: ./probe_0729_pi05_server.sh   (GPU=<n> to override, default 7)
# NOTE: pi05 FiLM runs (film_frombase/film_onnaive) are NOT covered — probing them
# needs film_contact_pi05 routing in the probes (different apply signature) plus
# their training env; naive is the generality claim.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
RV="$DIR/datasets/lges_case_pick_0729_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RUN="$DIR/outputs/pi05_naive_0729"

[[ -d "$RV/meta" && -d "$RUN/checkpoints" ]] || { echo "[pi05] inputs missing" >&2; exit 1; }
mkdir -p "$DIR/probes"

for ck in best last; do
  CK="$RUN/checkpoints/$ck"
  [[ -e "$CK" ]] || { echo "[pi05] skip $ck (missing)"; continue; }
  state() { # <outfile> <extra...>
    local out=$1; shift
    "$PY" "$DIR/probe_state_authority.py" --naive --all-episodes \
      --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" "$@" \
      > "$DIR/probes/$out" 2>&1
    echo "[pi05] $out rc=$? -> $(grep -a -m1 'ALL frames' "$DIR/probes/$out" | sed 's/^ *//')"
  }
  state "0729_state_pi05naive_${ck}_pc_fc.txt"  --swap firstcontact --pre-contact 10
  state "0729_state_pi05naive_${ck}_ramp8.txt"  --swap fcscale --fc-mag 8
  state "0729_state_pi05naive_${ck}_ramp12.txt" --swap fcscale --fc-mag 12
  state "0729_state_pi05naive_${ck}_pc_r12.txt" --swap fcscale --fc-mag 12 --pre-contact 10
  for off in 1 30; do
    "$PY" "$DIR/probe_press_sim.py" --naive --checkpoint "$CK" \
      --dataset-root "$RV" --repo-id "$REPO_VAL" \
      --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
      > "$DIR/probes/0729_sim_pi05naive_${ck}_off${off}_fzdelta.txt" 2>&1
    echo "[pi05] sim $ck off$off rc=$?"
  done
  "$PY" "$DIR/eval_offline.py" --checkpoint "$CK" \
    --val-root "$RV" --repo-id "$REPO_VAL" \
    > "$DIR/probes/0729_eval_pi05naive_${ck}.txt" 2>&1; echo "[pi05] eval $ck rc=$?"
done

echo; echo "===== pi05_naive verdict material ====="
grep -a -H "ALL frames" "$DIR"/probes/0729_state_pi05naive_*.txt
grep -a -H -A3 "── summary" "$DIR"/probes/0729_sim_pi05naive_*.txt | grep -v "^--$"
grep -a -H "OVERALL" "$DIR"/probes/0729_eval_pi05naive_*.txt
echo "[pi05] DONE — commit probes/ + push"
