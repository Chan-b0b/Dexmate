#!/usr/bin/env bash
# Extend the two finished MeanFlow runs (smolvla_meanflow_0708{,_abs}) from their
# 20k-step checkpoints to STEPS total (default 35000), concurrently on GPU 6.
#
# lerobot's cosine scheduler auto-scaled decay to 20k on the original runs (LR floored
# at 2.5e-6 by the end); resuming with steps=35000 rebuilds it over the full 30k decay
# horizon, so LR warm-restarts at ~2.8e-5 and decays to the floor by step 30k.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
export TOKENIZERS_PARALLELISM=false

STEPS="${STEPS:-35000}"

extend() {  # extend <run_name>
  local RUN="$1"
  local CFG="$DIR/outputs/$RUN/checkpoints/last/pretrained_model/train_config.json"
  local LOGDIR="$DIR/logs/$RUN"
  local LOG="$LOGDIR/train.log"
  if [[ ! -f "$CFG" ]]; then
    echo "[mf-extend] SKIP $RUN: no checkpoint config at $CFG" >&2
    return 1
  fi
  rm -f "$LOG.done"
  "$VENV/bin/python" "$DIR/tb_log.py" "$LOG" "$LOGDIR/tb" &
  "$VENV/bin/lerobot-train" \
    --config_path="$CFG" \
    --resume=true \
    --steps="$STEPS" \
    --dataset.discover_packages_path=smolvla_meanflow >> "$LOG" 2>&1
  local RC=$?
  touch "$LOG.done"
  echo "[mf-extend] $RUN finished rc=$RC"
  return $RC
}

echo "[mf-extend] extending both MeanFlow runs to $STEPS steps on GPU $CUDA_VISIBLE_DEVICES"
extend smolvla_meanflow_0708 &
PID_REL=$!
sleep 120
extend smolvla_meanflow_0708_abs &
PID_ABS=$!

REL_RC=0; wait "$PID_REL" || REL_RC=$?
ABS_RC=0; wait "$PID_ABS" || ABS_RC=$?
echo "[mf-extend] DONE  meanflow_0708=$REL_RC meanflow_0708_abs=$ABS_RC"
[[ "$REL_RC" == "0" && "$ABS_RC" == "0" ]]
