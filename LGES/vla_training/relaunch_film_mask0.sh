#!/usr/bin/env bash
# Fresh 30k FiLM-from-base runs with FILM_MASK_FORCE=0 (2026-07-09 decision:
# keep the wrench dims in the state, FiLM added ON TOP — no bottleneck).
# The earlier mask_force=1 runs were archived as outputs/*_mask1.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES=6
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

INPUT_FEATURES='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE_CKPT="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"

FILM_VARIANT=v2 FILM_MASK_FORCE=0 RUN_NAME=smolvla_film_0708 \
  INIT_CKPT="$BASE_CKPT" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708 \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708" \
  "$DIR/train_film.sh" --policy.input_features="$INPUT_FEATURES" --steps=30000 \
  > "$DIR/logs/orch_film_base_mask0.out" 2>&1 &
P1=$!

FILM_VARIANT=v2 FILM_MASK_FORCE=0 RUN_NAME=smolvla_film_0708_abs \
  INIT_CKPT="$BASE_CKPT" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708_abs \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs" \
  "$DIR/train_film.sh" --policy.input_features="$INPUT_FEATURES" --steps=30000 \
  > "$DIR/logs/orch_film_base_abs_mask0.out" 2>&1 &
P2=$!

echo "[film-mask0] delta pid=$P1 abs pid=$P2"
R1=0; wait "$P1" || R1=$?
echo "[film-mask0] delta finished rc=$R1"
R2=0; wait "$P2" || R2=$?
echo "[film-mask0] abs finished rc=$R2"
[[ "$R1" == 0 && "$R2" == 0 ]]
