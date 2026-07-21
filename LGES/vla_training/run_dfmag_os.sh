#!/usr/bin/env bash
# Contact-transition oversampling round: same winning dF setting (prefix+mask1,
# cond=contact,fz,seal,dfmag) + FILM_OVERSAMPLE_BOOST=10 — frames within +-5 of a
# >=2 N/frame |F| drop get 10x sampling weight (~40% of samples), so the 1-2
# frame contact dip actually gets gradient exposure (probe decomposition showed
# the plain dF run bound its stop to post-seal signals instead). GPU 4.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES=4
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # stored token invalid; anonymous works for public repos
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

IF16='{"observation.state": {"type": "STATE", "shape": [16]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  FILM_OVERSAMPLE_BOOST=10 \
  RUN_NAME=smolvla_film_0708_dF_prefix_mask1_os10 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708_dF \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_dF" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_dF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_dF_os10.out" 2>&1 &
P1=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  FILM_OVERSAMPLE_BOOST=10 \
  RUN_NAME=smolvla_film_0708_abs_dF_prefix_mask1_os10 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708_abs_dF \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs_dF" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs_dF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_abs_dF_os10.out" 2>&1 &
P2=$!

echo "[os10] rel pid=$P1 abs pid=$P2"
R1=0; wait $P1 || R1=$?; echo "[os10] rel finished rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[os10] abs finished rc=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
