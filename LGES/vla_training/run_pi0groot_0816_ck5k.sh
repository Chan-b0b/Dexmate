#!/usr/bin/env bash
# 0816 retrain (2026-08-20): π0 + GR00T × naive/film-state/film-layers, save EVERY 5k
# checkpoint and KEEP ALL of them (select_best runs WITHOUT --prune) so authority can be
# probed per-checkpoint over training (motivated by the best↔last divergence: SmolVLA
# layers strengthens with training while π0 film arms decay).
# Recipes = the 0816 round; only save_freq (10000->5000) and run names (_r2) differ.
# GPUs: 0 pi0_naive | 1 pi0_state | 2 pi0_layers | 3 groot_naive | 4 groot_state | 5 groot_layers
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0816
REPO_VAL_ID=Chanho-Lee/lges_case_pick_0816_val
RT="$DIR/datasets/lges_case_pick_0816"
RV="$DIR/datasets/lges_case_pick_0816_val"
FZ=1.4
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FB=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FZ"
    FILM_DATASET_ROOT="$RT")

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[ck5k] datasets missing" >&2; exit 1; }
mkdir -p "$DIR/logs"

PI0_ARGS=(--policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false
  --policy.gradient_checkpointing=true
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME"
  --batch_size=8 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=32)
GROOT_ARGS=(--policy.type=groot --policy.device=cuda --policy.push_to_hub=false
  --dataset.repo_id="$REPO" --dataset.root="$RT"
  --batch_size=8 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=32)

chain() { # name gpu inject entry argsvar
  local name=$1 gpu=$2 inject=$3 entry=$4 argsvar=$5
  local -n A="$argsvar"
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" FILM_INJECT="$inject" \
    "$PY" "$DIR/$entry" "${A[@]}" \
    --output_dir="$DIR/outputs/$name" --job_name="$name" \
    > "$DIR/logs/orch_${name}.out" 2>&1
  local rc=$?
  echo "[ck5k] $name train rc=$rc"
  [[ $rc == 0 ]] || return $rc
  # NO --prune: keep every 5k checkpoint; this just records val losses + best symlink
  env CUDA_VISIBLE_DEVICES="$gpu" "${FB[@]}" FILM_INJECT="$inject" \
    "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$name" \
    --val-root "$RV" --repo-id "$REPO_VAL_ID" >> "$DIR/logs/best_0816_r2.out" 2>&1
  echo "[ck5k] best($name): $(readlink "$DIR/outputs/$name/checkpoints/best" 2>/dev/null)"
}

chain pi0_naive_0816_r2       0 state  train_pi05.py       PI0_ARGS   & P1=$!
chain pi0_film_state_0816_r2  1 state  train_film_pi0.py   PI0_ARGS   & P2=$!
chain pi0_film_layers_0816_r2 2 layers train_film_pi0.py   PI0_ARGS   & P3=$!
chain groot_naive_0816_r2       3 state  train_groot.py      GROOT_ARGS & P4=$!
chain groot_film_state_0816_r2  4 state  train_film_groot.py GROOT_ARGS & P5=$!
chain groot_film_layers_0816_r2 5 layers train_film_groot.py GROOT_ARGS & P6=$!
echo "[ck5k] pids pi0=$P1/$P2/$P3 groot=$P4/$P5/$P6"
FAIL=0
for p in $P1 $P2 $P3 $P4 $P5 $P6; do wait "$p" || FAIL=$((FAIL + 1)); done
echo "[ck5k] DONE (failures: $FAIL)"
[[ $FAIL == 0 ]]
