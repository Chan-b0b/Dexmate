#!/usr/bin/env bash
# pi05 additions to the 0729 round: naive first (GPU5), then film-from-base + film_on_naive
# in parallel on separate GPUs (5 + 7) once naive's checkpoint exists. Each pi0.5 instance
# needs ~64GB -- MUST run on separate GPUs, not together (two on one GPU OOM'd once already).
# num_workers=32 (bumped from 16 -- SmolVLA runs showed 24+ workers needed to avoid a
# dataloader bottleneck; same fix applied here).
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

# ---- phase 1: naive on GPU5 -----------------------------------------------------
CUDA_VISIBLE_DEVICES=5 "$PY" "$DIR/train_pi05.py" \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/pi05_naive_0729" --job_name=pi05_naive_0729 \
  > "$DIR/logs/orch_pi05_naive_0729.out" 2>&1
R1=$?
echo "[pi05_0729] naive rc=$R1"

env CUDA_VISIBLE_DEVICES=5 "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/pi05_naive_0729" \
  --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_pi05_0729.out" 2>&1 \
  && echo "[pi05_0729] best(pi05_naive_0729): $(readlink "$DIR/outputs/pi05_naive_0729/checkpoints/best" 2>/dev/null)"

if [[ $R1 != 0 ]]; then
  echo "[pi05_0729] naive failed -- skipping FiLM phase (film_on_naive needs its checkpoint)"
  exit 1
fi
NAIVE_CKPT="$DIR/outputs/pi05_naive_0729/checkpoints/best/pretrained_model"

# ---- phase 2: film-from-base (GPU5) + film_on_naive (GPU7), parallel ----------
CUDA_VISIBLE_DEVICES=5 FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 \
  FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_pi05.py" \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/pi05_film_frombase_0729" --job_name=pi05_film_frombase_0729 \
  > "$DIR/logs/orch_pi05_film_frombase_0729.out" 2>&1 &
P2=$!

CUDA_VISIBLE_DEVICES=7 FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 \
  FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_pi05.py" \
  --policy.path="$NAIVE_CKPT" --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/pi05_film_onnaive_0729" --job_name=pi05_film_onnaive_0729 \
  > "$DIR/logs/orch_pi05_film_onnaive_0729.out" 2>&1 &
P3=$!

echo "[pi05_0729] pids frombase=$P2(GPU5) onnaive=$P3(GPU7)"
R2=0; wait $P2 || R2=$?; echo "[pi05_0729] frombase rc=$R2"
R3=0; wait $P3 || R3=$?; echo "[pi05_0729] onnaive rc=$R3"

best() { # run film_env...
  local run="$1"; shift
  env CUDA_VISIBLE_DEVICES=5 "$@" "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$run" \
    --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_pi05_0729.out" 2>&1 \
    && echo "[pi05_0729] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}
best pi05_film_frombase_0729 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"
best pi05_film_onnaive_0729 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"

echo "[pi05_0729] DONE naive=$R1 frombase=$R2 onnaive=$R3"
[[ $R1 == 0 && $R2 == 0 && $R3 == 0 ]]
