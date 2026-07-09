#!/usr/bin/env bash
# Train SmolVLA-MeanFlow (one-step action generation) on the two 0708 case_pick
# datasets, CONCURRENTLY on one GPU (default GPU 1, override with GPU=<idx>):
#   1. smolvla_meanflow_0708      — Chanho-Lee/lges_case_pick_0708      (action 7, delta)
#   2. smolvla_meanflow_0708_abs  — Chanho-Lee/lges_case_pick_0708_abs  (action 8, absolute)
#
# Both warm-start from ~/checkpoints/smolvla_meanflow_base (lerobot/smolvla_base
# converted with smolvla_meanflow/scripts/init_from_smolvla.py — exactly equivalent
# to smolvla_base at step 0). The policy plugin lives in ~/smolvla_meanflow and is
# loaded via --dataset.discover_packages_path (NOT --policy.*: lerobot re-parses
# --policy.* args as policy-config overrides, so the plugin flag must sit on
# another prefix). lerobot itself is unmodified.
#
# Mirrors train_smolvla.sh conventions: outputs/<run>, logs/<run>/train.log,
# tb_log.py tensorboard scalars beside the run.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/maverick/vla_venv
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
# The shared /data/cache/hf has an unreadable token file — use the per-user cache.
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
export TOKENIZERS_PARALLELISM=false

MF_CKPT="$HOME/checkpoints/smolvla_meanflow_base"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-32}"

RENAME_MAP='{"observation.images.head": "observation.images.camera1", "observation.images.head_depth": "observation.images.camera2"}'
INPUT_FEATURES='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'

launch() {  # launch <run_name> <repo_id> <dataset_root>
  local RUN="$1" REPO="$2" ROOT="$3"
  local OUT="$DIR/outputs/$RUN"
  local LOGDIR="$DIR/logs/$RUN"
  local LOG="$LOGDIR/train.log"
  if [[ -e "$OUT" ]]; then
    echo "[mf-orch] SKIP $RUN: output dir $OUT already exists (use train_smolvla.sh --resume conventions)" >&2
    return 1
  fi
  mkdir -p "$LOGDIR"
  rm -f "$LOG.done"
  "$VENV/bin/python" "$DIR/tb_log.py" "$LOG" "$LOGDIR/tb" &
  echo "[mf-orch] $RUN: tensorboard --logdir $LOGDIR/tb"

  "$VENV/bin/lerobot-train" \
    --policy.path="$MF_CKPT" \
    --dataset.discover_packages_path=smolvla_meanflow \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO" \
    --dataset.root="$ROOT" \
    --rename_map="$RENAME_MAP" \
    --policy.input_features="$INPUT_FEATURES" \
    --batch_size="$BATCH" \
    --steps="$STEPS" \
    --save_freq=2000 \
    --log_freq=50 \
    --num_workers=12 \
    --output_dir="$OUT" \
    --job_name="$RUN" > "$LOG" 2>&1
  local RC=$?
  touch "$LOG.done"
  echo "[mf-orch] $RUN finished rc=$RC"
  return $RC
}

echo "[mf-orch] launching both MeanFlow runs on GPU $CUDA_VISIBLE_DEVICES (steps=$STEPS batch=$BATCH)"

launch smolvla_meanflow_0708     Chanho-Lee/lges_case_pick_0708     "$DIR/datasets/lges_case_pick_0708" &
PID_REL=$!
sleep 120   # stagger: let run 1 allocate before run 2 piles on
launch smolvla_meanflow_0708_abs Chanho-Lee/lges_case_pick_0708_abs "$DIR/datasets/lges_case_pick_0708_abs" &
PID_ABS=$!

REL_RC=0; wait "$PID_REL" || REL_RC=$?
ABS_RC=0; wait "$PID_ABS" || ABS_RC=$?

echo "[mf-orch] DONE  meanflow_0708=$REL_RC meanflow_0708_abs=$ABS_RC"
echo "[mf-orch] outputs: $DIR/outputs/{smolvla_meanflow_0708,smolvla_meanflow_0708_abs}"
[[ "$REL_RC" == "0" && "$ABS_RC" == "0" ]]
