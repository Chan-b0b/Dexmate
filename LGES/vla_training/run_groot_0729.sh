#!/usr/bin/env bash
# 0729 GR00T N1.5 round (2026-08-14): naive + FiLM-state in parallel — the 4th
# architecture for the state-token injection story (SmolVLA prefix / π0 state / ACT
# encoder / GR00T action-head state_encoder, film_contact_groot.py).
#
#   GPU $GPU_NAIVE : groot_naive_0729       (nvidia/GR00T-N1.5-3B finetune, no FiLM)
#   GPU $GPU_FILM  : groot_film_state_0729  (state-token FiLM, same finetune recipe)
#
# GR00T's lerobot port freezes the Eagle VLM by default (tune_llm/visual=False,
# tune_projector/diffusion=True) — both arms train the action head only. FiLM
# calibration matches the pi0/pi05/ACT rounds (cond=contact,fz,seal mask_force=1
# FZ_OFF=2.1, F0/TAU/FZ_TAU=6/4/5). GR00T is min-max-normalized: c-hat stats come from
# load_state_minmax, NOT film_contact's mean/std loaders. No rename_map — the groot
# processor packs all observation.images.* keys into its video tensor as-is.
#
#   ./run_groot_0729.sh                      # GPUs 4 + 6
#   GPU_NAIVE=0 GPU_FILM=1 ./run_groot_0729.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_NAIVE="${GPU_NAIVE:-4}"
GPU_FILM="${GPU_FILM:-6}"

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
FILM_FZ_OFF=2.1   # from logs/validate_0729.out FZ_MEDIAN, same as the other rounds

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[groot_0729] datasets missing under $DIR/datasets" >&2; exit 1; }
mkdir -p "$DIR/logs"

CUDA_VISIBLE_DEVICES="$GPU_NAIVE" "$PY" "$DIR/train_groot.py" \
  --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/groot_naive_0729" --job_name=groot_naive_0729 \
  > "$DIR/logs/orch_groot_naive_0729.out" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES="$GPU_FILM" env FILM_VARIANT=v2 FILM_COND=contact,fz,seal \
  FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_groot.py" \
  --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/groot_film_state_0729" --job_name=groot_film_state_0729 \
  > "$DIR/logs/orch_groot_film_state_0729.out" 2>&1 &
P2=$!
echo "[groot_0729] pids naive=$P1(GPU$GPU_NAIVE) film=$P2(GPU$GPU_FILM)"

best() { # run gpu film_env...
  local run="$1" gpu="$2"; shift 2
  env CUDA_VISIBLE_DEVICES="$gpu" "$@" "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$run" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_groot_0729.out" 2>&1 \
    && echo "[groot_0729] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}

R1=0; wait $P1 || R1=$?; echo "[groot_0729] naive rc=$R1"
[[ $R1 == 0 ]] && best groot_naive_0729 "$GPU_NAIVE"
R2=0; wait $P2 || R2=$?; echo "[groot_0729] film rc=$R2"
[[ $R2 == 0 ]] && best groot_film_state_0729 "$GPU_FILM" \
  FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"

echo "[groot_0729] DONE naive=$R1 film=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
