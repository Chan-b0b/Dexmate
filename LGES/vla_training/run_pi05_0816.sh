#!/usr/bin/env bash
# 0816 round, pi05 extension (2026-08-18): naive / film-suffix (the token arm — π0.5 has
# no state token) / film-layers on Chanho-Lee/lges_case_pick_0816. Recipes = the 0729
# pi05 round (bs8 50k save10k grad-ckpt, pi0-style rename) with num_workers=32 and
# FILM_FZ_OFF=1.4. Per arm: train -> select_best (--prune) -> battery
# (run_battery_v2.sh, PFX=0816).
#   GPU $GPU_NAIVE(7) naive | $GPU_SUFFIX(0) film-suffix | $GPU_LAYERS(1) film-layers
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_NAIVE="${GPU_NAIVE:-7}"
GPU_SUFFIX="${GPU_SUFFIX:-0}"
GPU_LAYERS="${GPU_LAYERS:-1}"

REPO=Chanho-Lee/lges_case_pick_0816
REPO_VAL_ID=Chanho-Lee/lges_case_pick_0816_val
RT="$DIR/datasets/lges_case_pick_0816"
RV="$DIR/datasets/lges_case_pick_0816_val"
FZ=1.4
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FB=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FZ"
    FILM_DATASET_ROOT="$RT")

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[pi05_0816] datasets missing" >&2; exit 1; }
mkdir -p "$DIR/logs" "$DIR/probes"

TRAIN_ARGS=(--policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false
  --policy.gradient_checkpointing=true
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME"
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32)

chain() { # name gpu kind inject entry_script
  local name=$1 gpu=$2 kind=$3 inject=$4 entry=$5
  CUDA_VISIBLE_DEVICES="$gpu" env "${FB[@]}" FILM_INJECT="$inject" \
    "$PY" "$DIR/$entry" "${TRAIN_ARGS[@]}" \
    --output_dir="$DIR/outputs/$name" --job_name="$name" \
    > "$DIR/logs/orch_${name}.out" 2>&1
  local rc=$?
  echo "[pi05_0816] $name train rc=$rc"
  [[ $rc == 0 ]] || return $rc
  env CUDA_VISIBLE_DEVICES="$gpu" "${FB[@]}" FILM_INJECT="$inject" \
    "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$name" \
    --val-root "$RV" --repo-id "$REPO_VAL_ID" --prune >> "$DIR/logs/best_0816.out" 2>&1
  echo "[pi05_0816] best($name): $(readlink "$DIR/outputs/$name/checkpoints/best" 2>/dev/null)"
  VAL_ROOT="$RV" REPO_VAL="$REPO_VAL_ID" FZ_OFF="$FZ" PFX=0816 \
    "$DIR/run_battery_v2.sh" "$name" "$kind" "$inject" \
    "$DIR/outputs/$name/checkpoints/best" "$gpu" >> "$DIR/logs/battery_0816.out" 2>&1
  echo "[pi05_0816] battery($name) rc=$?"
}

chain pi05_naive_0816       "$GPU_NAIVE"  naive     suffix train_pi05.py      & P1=$!
chain pi05_film_suffix_0816 "$GPU_SUFFIX" film-pi05 suffix train_film_pi05.py & P2=$!
chain pi05_film_layers_0816 "$GPU_LAYERS" film-pi05 layers train_film_pi05.py & P3=$!
echo "[pi05_0816] pids naive=$P1(GPU$GPU_NAIVE) suffix=$P2(GPU$GPU_SUFFIX) layers=$P3(GPU$GPU_LAYERS)"
FAIL=0
for p in $P1 $P2 $P3; do wait "$p" || FAIL=$((FAIL + 1)); done
echo "[pi05_0816] DONE (failures: $FAIL)"
[[ $FAIL == 0 ]]
