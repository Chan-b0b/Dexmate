#!/usr/bin/env bash
# Relaunch the three 0721 FiLM runs with force thresholds CALIBRATED to the new
# robot (measured: hover |F| p50=4.8N, sealed p50=8.4N, contact jump +2.6N/frame;
# unsealed/sealed overlap heavily -> static 'contact' is weak here, dfmag is the
# reliable signal):  FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5   (F0=12 default would
# leave the contact channel ~always 0 on this robot). Values are baked into
# checkpoint buffers, so deploy reads them from the checkpoint. GPU 7.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES=7
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
export FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5

REPO=Chanho-Lee/lges_case_pick_0721
ROOT="$DIR/datasets/lges_case_pick_0721"
ROOT_DF="$DIR/datasets/lges_case_pick_0721_dF"
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
IF16='{"observation.state": {"type": "STATE", "shape": [16]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE="$("$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"

FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0721_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO="$REPO" DATASET_ROOT="$ROOT" FILM_DATASET_ROOT="$ROOT" \
  "$DIR/train_film.sh" --policy.input_features="$IF15" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0721_prefix_mask1.out" 2>&1 &
P2=$!
FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0721_dF_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0721_dF DATASET_ROOT="$ROOT_DF" FILM_DATASET_ROOT="$ROOT_DF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0721_dF.out" 2>&1 &
P3=$!
FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  FILM_OVERSAMPLE_BOOST=10 \
  RUN_NAME=smolvla_film_0721_dF_prefix_mask1_os10 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0721_dF DATASET_ROOT="$ROOT_DF" FILM_DATASET_ROOT="$ROOT_DF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0721_dF_os10.out" 2>&1 &
P4=$!

echo "[0721-cal] pids film=$P2 dF=$P3 dF_os10=$P4 (F0=6 tau=4 fz_tau=5)"
R2=0; wait $P2 || R2=$?; echo "[0721-cal] film_prefix_mask1 rc=$R2"
R3=0; wait $P3 || R3=$?; echo "[0721-cal] film_dF rc=$R3"
R4=0; wait $P4 || R4=$?; echo "[0721-cal] film_dF_os10 rc=$R4"
echo "[0721-cal] DONE film=$R2 dF=$R3 dF_os10=$R4"
[[ $R2 == 0 && $R3 == 0 && $R4 == 0 ]]
