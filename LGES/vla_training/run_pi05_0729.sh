#!/usr/bin/env bash
# pi05 additions to the 0729 round (naive on GPU7 + FiLM-pi05 suffix/mask1 on GPU2).
# Mirrors pi05_naive_0721_0727's config (bs8, grad-ckpt, 50k) and the winning
# FiLM setting (cond=contact,fz,seal, mask_force=1) from the 0729 SmolVLA runs.
# Dataset only has 2 real cameras (head, head_depth); pi05_base's 3rd image
# slot (right_wrist_0_rgb) is left unmapped, same as the 0721_0727 pi05 run.
# Each pi0.5 instance needs ~64GB -- MUST run on separate GPUs, not together
# (two on one GPU OOM'd and killed a completed 50k run once already).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FILM_FZ_OFF=2.1   # from logs/validate_0729.out FZ_MEDIAN

mkdir -p "$DIR/logs"

CUDA_VISIBLE_DEVICES=7 "$PY" "$DIR/train_pi05.py" \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/pi05_naive_0729" --job_name=pi05_naive_0729 \
  > "$DIR/logs/orch_pi05_naive_0729.out" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=2 FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF" \
  FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_pi05.py" \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/pi05_film_0729_suffix_mask1" --job_name=pi05_film_0729_suffix_mask1 \
  > "$DIR/logs/orch_pi05_film_0729_sm1.out" 2>&1 &
P2=$!

echo "[pi05_0729] pids naive=$P1 film_sm1=$P2"
R1=0; wait $P1 || R1=$?; echo "[pi05_0729] naive rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[pi05_0729] film_sm1 rc=$R2"

best() { # run film_env...
  local run="$1"; shift
  env CUDA_VISIBLE_DEVICES=7 "$@" "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$run" \
    --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_pi05_0729.out" 2>&1 \
    && echo "[pi05_0729] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}
best pi05_naive_0729
best pi05_film_0729_suffix_mask1 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"

echo "[pi05_0729] DONE naive=$R1 film_sm1=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
