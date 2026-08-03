#!/usr/bin/env bash
# 0729 RECAL round (2026-07-31): recalibrated FiLM channels from the measured 0729-train
# force distributions (hover |F| p90=5.4 < pre-seal p10=5.4; fz p90=2.7 < p10=3.2, the
# cleanest separator; fy rejected — static mount bias, no contact information):
#   contact = clip((|F|-5.5)/1)   (was F0=6/tau=4 -> sealed read only ~0.15)
#   fmag    = (|F|-5.5)/1         (graded |F|, NEW channel this round)
#   fz      = (fz-3.0)/0.7        (was off=2.1/tau=5 -> hover/press delta only 0.3)
#   seal    unchanged
# Two runs on GPU 7: prefix_mask1_recal (method) + prefix_mask0_recal (control).
# Naive is NOT retrained — FiLM calibration does not affect it; compare against
# smolvla_naive_0729. Probes vs state-swap matrix follow training automatically.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=7
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'

# recalibrated FiLM env — the single source of truth for this round
RECAL_ENV=(FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix
           FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1
           FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7)
# canonical channel order is (contact, fz, fmag, seal); measured c-hat anchors:
#   hover    = [0,    -1.43, -0.9, 0]
#   pre-seal = [0.4,   0.86,  0.4, 0]
#   sealed   = [1,     2.29,  1.2, 1]

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[recal] datasets missing" >&2; exit 1; }
BASE="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"
mkdir -p "$DIR/logs" "$DIR/probes"

film() { # run_name mask log_tag
  env "${RECAL_ENV[@]}" FILM_VARIANT=v2 FILM_MASK_FORCE="$2" \
    RUN_NAME="$1" INIT_CKPT="$BASE" \
    DATASET_REPO="$REPO" DATASET_ROOT="$RT" FILM_DATASET_ROOT="$RT" \
    "$DIR/train_film.sh" --policy.input_features="$IF15" --steps=50000 --save_freq=5000 \
    --policy.scheduler_decay_steps=50000 \
    > "$DIR/logs/orch_$3.out" 2>&1
}

film smolvla_film_0729_prefix_mask1_recal 1 film_0729_pm1_recal & P1=$!
film smolvla_film_0729_prefix_mask0_recal 0 film_0729_pm0_recal & P2=$!
echo "[recal] pids pm1=$P1 pm0=$P2"
R1=0; wait $P1 || R1=$?; echo "[recal] pm1_recal rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[recal] pm0_recal rc=$R2"
[[ -d "$HOME/outputs" ]] && echo "[recal] WARNING: stray \$HOME/outputs appeared!" >&2

for run_mask in "smolvla_film_0729_prefix_mask1_recal 1" "smolvla_film_0729_prefix_mask0_recal 0"; do
  set -- $run_mask; run=$1; mask=$2; tag=pm${mask}_recal
  env "${RECAL_ENV[@]}" FILM_MASK_FORCE="$mask" \
    "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$run" \
    --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_0729_recal.out" 2>&1
  echo "[recal] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
  # c-hat authority probes: std (0->1) + measured hover->pre-seal + hover->sealed
  env "${RECAL_ENV[@]}" FILM_MASK_FORCE="$mask" \
    "$PY" "$DIR/probe_film_authority.py" --checkpoint "$DIR/outputs/$run/checkpoints/best" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
    > "$DIR/probes/0729_${tag}_std.txt" 2>&1; echo "[recal] probe ${tag}_std rc=$?"
  env "${RECAL_ENV[@]}" FILM_MASK_FORCE="$mask" \
    "$PY" "$DIR/probe_film_authority.py" --checkpoint "$DIR/outputs/$run/checkpoints/best" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
    --c0 "0,-1.43,-0.9,0" --c1 "0.4,0.86,0.4,0" \
    > "$DIR/probes/0729_${tag}_preseal.txt" 2>&1; echo "[recal] probe ${tag}_preseal rc=$?"
  env "${RECAL_ENV[@]}" FILM_MASK_FORCE="$mask" \
    "$PY" "$DIR/probe_film_authority.py" --checkpoint "$DIR/outputs/$run/checkpoints/best" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
    --c0 "0,-1.43,-0.9,0" --c1 "1,2.29,1.2,1" \
    > "$DIR/probes/0729_${tag}_sealed.txt" 2>&1; echo "[recal] probe ${tag}_sealed rc=$?"
  # state-swap matrix (both/wrench/seal/onset)
  for sw in both wrench seal onset; do
    env "${RECAL_ENV[@]}" FILM_MASK_FORCE="$mask" \
      "$PY" "$DIR/probe_state_authority.py" --swap "$sw" \
      --checkpoint "$DIR/outputs/$run/checkpoints/best" \
      --dataset-root "$RV" --repo-id "$REPO_VAL" \
      > "$DIR/probes/0729_state_${tag}_$sw.txt" 2>&1
    echo "[recal] probe state_${tag}_$sw rc=$?"
  done
done
grep -a -H -A5 "── verdict" "$DIR"/probes/0729_*recal*.txt | grep -v "^--$"
echo "[recal] DONE pm1=$R1 pm0=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
