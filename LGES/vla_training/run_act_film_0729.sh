#!/usr/bin/env bash
# FiLM-ACT additions to the 0729 round: encoder/prefix-style state-token injection
# (film_contact_act.py), two init modes in parallel on separate GPUs:
#   - act_film_scratch_0729:  fresh ACT init (--policy.type=act), FiLM learned jointly
#     with the base policy from scratch -- the "film-from-base" analogue (ACT has no
#     pretrained base to start from; naive IS from-scratch here).
#   - act_film_onnaive_0729:  init from the already-converged act_0729 checkpoint
#     (--policy.path=...), FiLM added on top of a converged policy.
# cond=contact,fz,seal / mask_force=1, matching the winning SmolVLA recipe.
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
FILM_FZ_OFF=2.1   # from logs/validate_0729.out FZ_MEDIAN
NAIVE_CKPT="$DIR/outputs/act_0729/checkpoints/best/pretrained_model"

mkdir -p "$DIR/logs"

CUDA_VISIBLE_DEVICES=4 FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 \
  FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_act.py" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=32 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/act_film_scratch_0729" --job_name=act_film_scratch_0729 \
  > "$DIR/logs/orch_act_film_scratch_0729.out" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES=6 FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 \
  FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_act.py" \
  --policy.path="$NAIVE_CKPT" --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=32 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/act_film_onnaive_0729" --job_name=act_film_onnaive_0729 \
  > "$DIR/logs/orch_act_film_onnaive_0729.out" 2>&1 &
P2=$!

echo "[act_film_0729] pids scratch=$P1(GPU4) onnaive=$P2(GPU6)"
R1=0; wait $P1 || R1=$?; echo "[act_film_0729] scratch rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[act_film_0729] onnaive rc=$R2"

best() { # run film_env...
  local run="$1"; shift
  env CUDA_VISIBLE_DEVICES=4 "$@" "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$run" \
    --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_act_film_0729.out" 2>&1 \
    && echo "[act_film_0729] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}
best act_film_scratch_0729 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"
best act_film_onnaive_0729 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"

echo "[act_film_0729] DONE scratch=$R1 onnaive=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
