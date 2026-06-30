#!/usr/bin/env bash
# Fine-tune SmolVLA on the prev-action-augmented dataset (contact_aware_vla experiment).
# Mirrors LGES/vla_training/train_smolvla.sh but points at the NEW dataset and writes
# outputs under Research/, leaving vla_training untouched. tb_log.py is reused read-only.
#
# Fresh run:  RUN_NAME=smolvla_prevaction ./train_prevaction.sh [extra lerobot-train overrides]
#   e.g. a faster first look: RUN_NAME=smolvla_prevaction ./train_prevaction.sh --steps=20000
# Don't train while the robot demo is running — they share the GPU.
set -euo pipefail

VENV=/home/dexmate/vla_venv
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_TRAIN="$HERE/../../LGES/vla_training"
RUN="${RUN_NAME:-smolvla_prevaction}"
OUT="$HERE/outputs/$RUN"            # lerobot CREATES this; refuses if it exists.
LOGDIR="$HERE/logs/$RUN"            # logs+tb live BESIDE it, not inside.
LOG="$LOGDIR/train.log"
mkdir -p "$LOGDIR"

# TensorBoard via the existing tb_log.py (tails stdout -> TB scalars), read-only reuse.
rm -f "$LOG.done"
"$VENV/bin/python" "$VLA_TRAIN/tb_log.py" "$LOG" "$LOGDIR/tb" &
TB_PID=$!
trap 'touch "$LOG.done"; wait "$TB_PID" 2>/dev/null || true' EXIT
echo "[train] TensorBoard scalars -> $LOGDIR/tb   (tensorboard --logdir $LOGDIR/tb)"

# Same recipe as the baseline (fine-tune from smolvla_base, depth+head cameras),
# only the dataset differs. State is 22-dim; SmolVLA pads to max_state_dim=32, so
# no model change — the previously-zero state-projection slots now carry the
# previous action and get trained.
"$VENV/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/lges_suction_prevaction \
  --dataset.root="$HERE/datasets/lges_suction_prevaction" \
  --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.head_depth": "observation.images.camera2"}' \
  --batch_size=32 \
  --steps=60000 \
  --save_freq=2000 \
  --log_freq=50 \
  --num_workers=12 \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  "$@" 2>&1 | tee "$LOG"




# cd /home/dexmate/CNS_code/Dexmate/Research/contact_aware_vla
# /home/dexmate/vla_venv/bin/lerobot-train \
#   --config_path=outputs/smolvla_prevaction_dagger1/checkpoints/last/pretrained_model/train_config.json \
#   --resume=true \
#   --steps=30000