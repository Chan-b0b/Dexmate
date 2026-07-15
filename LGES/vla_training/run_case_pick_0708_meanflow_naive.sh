#!/usr/bin/env bash
# MeanFlow training warm-started from the TASK-ADAPTED naive checkpoints (instead of
# smolvla_base), concurrently on one GPU (default 6, override GPU=<idx>):
#   1. smolvla_meanflow_naive_0708      <- converted smolvla_naive_0708      (delta actions)
#   2. smolvla_meanflow_naive_0708_abs  <- converted smolvla_naive_0708_abs  (absolute actions)
#
# Rationale: the base-warm-started MeanFlow runs had to learn the task AND the average-
# velocity field at once (abs ended 5.5mm vs naive 3.4mm). Converting a task-adapted FM
# checkpoint is lossless (zero-init interval proj => u(x,t,t) == naive's v(x,t) at step 0),
# so training only distills the already-correct velocity field into average velocities.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/maverick/vla_venv
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
export TOKENIZERS_PARALLELISM=false

STEPS="${STEPS:-20000}"
BATCH="${BATCH:-32}"
GPU_IDX="${GPU:-6}"

RENAME_MAP='{"observation.images.head": "observation.images.camera1", "observation.images.head_depth": "observation.images.camera2"}'

convert() {  # convert <naive_run> <out_ckpt_dir>
  local SRC="$DIR/outputs/$1/checkpoints/last/pretrained_model" OUT="$2"
  if [[ -f "$OUT/model.safetensors" ]]; then
    echo "[mf-naive] conversion exists: $OUT"
    return 0
  fi
  # Force CPU: the source config says cuda and would otherwise land on GPU 0.
  CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$DIR/smolvla_meanflow/scripts/init_from_smolvla.py" \
    --src "$SRC" --out "$OUT" --num-steps 1
}

launch() {  # launch <run_name> <ckpt> <repo_id> <dataset_root>
  local RUN="$1" CKPT="$2" REPO="$3" ROOT="$4"
  local OUT="$DIR/outputs/$RUN" LOGDIR="$DIR/logs/$RUN" LOG="$DIR/logs/$RUN/train.log"
  if [[ -e "$OUT" ]]; then
    echo "[mf-naive] SKIP $RUN: $OUT already exists" >&2
    return 1
  fi
  mkdir -p "$LOGDIR"
  rm -f "$LOG.done"
  "$VENV/bin/python" "$DIR/tb_log.py" "$LOG" "$LOGDIR/tb" &
  echo "[mf-naive] $RUN: tensorboard --logdir $LOGDIR/tb"
  CUDA_VISIBLE_DEVICES="$GPU_IDX" "$VENV/bin/lerobot-train" \
    --policy.path="$CKPT" \
    --dataset.discover_packages_path=smolvla_meanflow \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT" \
    --rename_map="$RENAME_MAP" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --save_freq=2000 \
    --log_freq=50 \
    --num_workers=12 \
    --output_dir="$OUT" \
    --job_name="$RUN" > "$LOG" 2>&1
  local RC=$?
  touch "$LOG.done"
  echo "[mf-naive] $RUN finished rc=$RC"
  return $RC
}

CKPT_REL="$HOME/checkpoints/smolvla_meanflow_naive_0708"
CKPT_ABS="$HOME/checkpoints/smolvla_meanflow_naive_0708_abs"

echo "[mf-naive] converting naive checkpoints -> meanflow warm starts"
convert smolvla_naive_0708     "$CKPT_REL" || exit 1
convert smolvla_naive_0708_abs "$CKPT_ABS" || exit 1

echo "[mf-naive] launching both runs on GPU $GPU_IDX (steps=$STEPS batch=$BATCH)"
launch smolvla_meanflow_naive_0708     "$CKPT_REL" Chanho-Lee/lges_case_pick_0708     "$DIR/datasets/lges_case_pick_0708" &
PID_REL=$!
sleep 120
launch smolvla_meanflow_naive_0708_abs "$CKPT_ABS" Chanho-Lee/lges_case_pick_0708_abs "$DIR/datasets/lges_case_pick_0708_abs" &
PID_ABS=$!

REL_RC=0; wait "$PID_REL" || REL_RC=$?
ABS_RC=0; wait "$PID_ABS" || ABS_RC=$?
echo "[mf-naive] DONE  rel=$REL_RC abs=$ABS_RC"
[[ "$REL_RC" == "0" && "$ABS_RC" == "0" ]]
