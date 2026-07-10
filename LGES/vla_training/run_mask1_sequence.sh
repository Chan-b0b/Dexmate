#!/usr/bin/env bash
# mask_force=1 round (probe showed mask0 loses c-hat authority; mask1 is the
# only setting with right-sign contact gating):
#   Stage 1: resume smolvla_film_0708_mask1{,_abs} (suffix) 10k -> 30k
#   Stage 2: fresh smolvla_film_0708_prefix_mask1{,_abs} (prefix) 30k
# Sequential stages, two runs parallel within each. num_workers=24,
# save_freq=4000 (disk is a shared 879G volume that already filled once).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=6
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

RD="$DIR/datasets/lges_case_pick_0708";     PD=Chanho-Lee/lges_case_pick_0708
RA="$DIR/datasets/lges_case_pick_0708_abs"; PA=Chanho-Lee/lges_case_pick_0708_abs
INPUT_FEATURES='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'

prune_ts() { # keep only the training_state that 'last' points to
  local ck="$DIR/outputs/$1/checkpoints"; [[ -d "$ck" ]] || return 0
  local keep; keep="$(readlink -f "$ck/last" 2>/dev/null | xargs -r basename)"
  for c in "$ck"/0*/; do
    [[ "$(basename "$c")" == "$keep" ]] || rm -rf "$c/training_state"
  done
}

echo "[mask1-seq] STAGE 1: resume mask1 suffix pair 10k -> 30k (nw=24)"
FILM_VARIANT=v2 FILM_INJECT=suffix FILM_MASK_FORCE=1 FILM_DATASET_ROOT="$RD" "$PY" "$DIR/train_film.py" \
  --config_path="$DIR/outputs/smolvla_film_0708_mask1/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --steps=30000 --num_workers=24 --save_freq=4000 \
  > "$DIR/logs/orch_film_mask1_30k.out" 2>&1 &
P1=$!
FILM_VARIANT=v2 FILM_INJECT=suffix FILM_MASK_FORCE=1 FILM_DATASET_ROOT="$RA" "$PY" "$DIR/train_film.py" \
  --config_path="$DIR/outputs/smolvla_film_0708_abs_mask1/checkpoints/last/pretrained_model/train_config.json" \
  --resume=true --steps=30000 --num_workers=24 --save_freq=4000 \
  > "$DIR/logs/orch_film_abs_mask1_30k.out" 2>&1 &
P2=$!
R1=0; wait $P1 || R1=$?; echo "[mask1-seq] mask1 suffix delta rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[mask1-seq] mask1 suffix abs rc=$R2"
prune_ts smolvla_film_0708_mask1; prune_ts smolvla_film_0708_abs_mask1
df -h /home/maverick | tail -1

echo "[mask1-seq] STAGE 2: fresh prefix+mask1 pair 30k (nw=24)"
BASE_CKPT="$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"
FILM_VARIANT=v2 FILM_INJECT=prefix FILM_MASK_FORCE=1 RUN_NAME=smolvla_film_0708_prefix_mask1 \
  INIT_CKPT="$BASE_CKPT" DATASET_REPO="$PD" DATASET_ROOT="$RD" FILM_DATASET_ROOT="$RD" \
  "$DIR/train_film.sh" --policy.input_features="$INPUT_FEATURES" --steps=30000 --num_workers=24 --save_freq=4000 \
  > "$DIR/logs/orch_film_prefix_mask1.out" 2>&1 &
P3=$!
FILM_VARIANT=v2 FILM_INJECT=prefix FILM_MASK_FORCE=1 RUN_NAME=smolvla_film_0708_abs_prefix_mask1 \
  INIT_CKPT="$BASE_CKPT" DATASET_REPO="$PA" DATASET_ROOT="$RA" FILM_DATASET_ROOT="$RA" \
  "$DIR/train_film.sh" --policy.input_features="$INPUT_FEATURES" --steps=30000 --num_workers=24 --save_freq=4000 \
  > "$DIR/logs/orch_film_abs_prefix_mask1.out" 2>&1 &
P4=$!
R3=0; wait $P3 || R3=$?; echo "[mask1-seq] prefix_mask1 delta rc=$R3"
R4=0; wait $P4 || R4=$?; echo "[mask1-seq] prefix_mask1 abs rc=$R4"
prune_ts smolvla_film_0708_prefix_mask1; prune_ts smolvla_film_0708_abs_prefix_mask1

echo "[mask1-seq] DONE suffix_mask1=$R1/$R2 prefix_mask1=$R3/$R4"
[[ $R1 == 0 && $R2 == 0 && $R3 == 0 && $R4 == 0 ]]
