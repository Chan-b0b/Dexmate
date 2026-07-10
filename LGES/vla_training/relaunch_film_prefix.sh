#!/usr/bin/env bash
# Inject-point ablation: FiLM-from-base 30k with FILM_INJECT=prefix (state-token
# modulation in VLM space) vs the concurrently-running suffix runs. Same
# everything else: v2, cond=contact,fz,seal, mask_force=0.
# NB: eval must use the SAME cond+inject+mask_force as training.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES=6
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

INPUT_FEATURES='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE_CKPT="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"

FILM_VARIANT=v2 FILM_INJECT=prefix FILM_MASK_FORCE=0 RUN_NAME=smolvla_film_0708_prefix \
  INIT_CKPT="$BASE_CKPT" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708 \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708" \
  "$DIR/train_film.sh" --policy.input_features="$INPUT_FEATURES" --steps=30000 \
  > "$DIR/logs/orch_film_base_prefix.out" 2>&1 &
P1=$!

FILM_VARIANT=v2 FILM_INJECT=prefix FILM_MASK_FORCE=0 RUN_NAME=smolvla_film_0708_abs_prefix \
  INIT_CKPT="$BASE_CKPT" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0708_abs \
  DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs" \
  FILM_DATASET_ROOT="$DIR/datasets/lges_case_pick_0708_abs" \
  "$DIR/train_film.sh" --policy.input_features="$INPUT_FEATURES" --steps=30000 \
  > "$DIR/logs/orch_film_base_abs_prefix.out" 2>&1 &
P2=$!

echo "[film-prefix] delta pid=$P1 abs pid=$P2"
R1=0; wait "$P1" || R1=$?
echo "[film-prefix] delta finished rc=$R1"
R2=0; wait "$P2" || R2=$?
echo "[film-prefix] abs finished rc=$R2"
[[ "$R1" == 0 && "$R2" == 0 ]]
