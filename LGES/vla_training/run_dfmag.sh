#!/usr/bin/env bash
# dF round: FiLM v2, cond=contact,fz,seal,dfmag (d|F|/dt channel — payload-robust
# contact transient), inject=prefix, mask_force=1 (the winning 0708 setting), 30k,
# on the derived *_dF datasets (state 16). GPU 4. rel + abs in parallel.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES=4
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

IF16='{"observation.state": {"type": "STATE", "shape": [16]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0708_dF_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708_dF \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_dF" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_dF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_dF_prefix_mask1.out" 2>&1 &
P1=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0708_abs_dF_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708_abs_dF \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs_dF" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs_dF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_abs_dF_prefix_mask1.out" 2>&1 &
P2=$!

echo "[dfmag] rel pid=$P1 abs pid=$P2"
R1=0; wait $P1 || R1=$?; echo "[dfmag] rel finished rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[dfmag] abs finished rc=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
