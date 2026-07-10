#!/usr/bin/env bash
# Finetune a FiLM contact-conditioned SmolVLA (descend-until-contact, V1/V2) FROM the
# corrected vanilla checkpoint. The FiLM starts as identity, so training begins at the
# base policy's behavior and learns the contact modulation.
#
#   FILM_VARIANT=v2 RUN_NAME=film_v2 ./train_film.sh                 # full run
#   FILM_VARIANT=v1 RUN_NAME=film_v1 ./train_film.sh                 # decorrelated control
#   FILM_VARIANT=v2 RUN_NAME=smoke   ./train_film.sh --steps=4 --save_freq=2   # smoke test
#
# Don't train while the robot demo is running — they share the GPU.
set -euo pipefail

VENV="${VENV:-/home/dexmate/vla_venv}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${FILM_VARIANT:-v2}"
INIT="${INIT_CKPT:-$DIR/outputs/smolvla_20260624_081946/checkpoints/last/pretrained_model}"
DATASET_REPO="${DATASET_REPO:-local/lges_suction}"
DATASET_ROOT="${DATASET_ROOT:-$DIR/datasets/lges_suction}"
RUN="${RUN_NAME:-film_${VARIANT}_$(date +%Y%m%d_%H%M%S)}"
OUT="$DIR/outputs/$RUN"
LOGDIR="$DIR/logs/$RUN"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/train.log"

if [[ ! -d "$INIT" ]]; then
  echo "[train-film] init checkpoint not found: $INIT" >&2
  echo "             set INIT_CKPT=<dir with pretrained_model>" >&2
  exit 1
fi
echo "[train-film] variant=$VARIANT  init=$INIT  out=$OUT"

FILM_VARIANT="$VARIANT" "$VENV/bin/python" "$DIR/train_film.py" \
  --policy.path="$INIT" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.head_depth": "observation.images.camera2"}' \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=2000 \
  --log_freq=50 \
  --num_workers=32 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  "$@" 2>&1 | tee "$LOG"


#FILM_COND=contact,seal FILM_INJECT=suffix FILM_MASK_FORCE=0 FILM_VARIANT=v2 RUN_NAME=film_v2_contactseal_nomask_suffix ./train_film.sh