#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base, 450M) on the converted LGES
# suction dataset. Runs lerobot's standard training loop inside vla_venv
# (isolated venv that reuses the Jetson torch from /opt/venv).
#
# Fresh run:  RUN_NAME=myrun ./train_smolvla.sh [extra lerobot-train overrides...]
#   e.g.      RUN_NAME=myrun ./train_smolvla.sh --steps=40000
# Resume:     [RUN_NAME=myrun] ./train_smolvla.sh --resume   [--steps=N to extend]
#   continues outputs/<RUN_NAME> from its latest checkpoint, reusing the saved
#   config (dataset, hyperparams). RUN_NAME is optional: without it, resumes the
#   most recently trained run under outputs/ that has a checkpoint. Extra
#   overrides still apply (e.g. --steps to extend).
#
# Don't train while the robot demo is running — they share the GPU.
set -euo pipefail

# An explicit VENV wins; otherwise take whichever per-host venv exists — the x86
# training box (setup_venv.sh) or the Jetson. Same rule in train_film.sh.
VENV="${VENV:-$([[ -x /home/maverick/vla_venv/bin/python ]] \
  && echo /home/maverick/vla_venv || echo /home/dexmate/vla_venv)}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull our own --resume flag out of the args forwarded to lerobot-train.
RESUME=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--resume" ]]; then RESUME=1; else ARGS+=("$a"); fi
done

# Choose the run directory. A fresh run gets a timestamp name (or RUN_NAME).
# --resume must point at an EXISTING run, so never invent a timestamp there:
# use RUN_NAME if given, otherwise auto-pick the most recently trained run
# under outputs/ that has a saved checkpoint.
if [[ -n "${RUN_NAME:-}" ]]; then
  RUN="$RUN_NAME"
elif [[ "$RESUME" == "1" ]]; then
  RUN=""
  for d in $(ls -1dt "$DIR"/outputs/*/ 2>/dev/null); do
    if [[ -f "${d}checkpoints/last/pretrained_model/train_config.json" ]]; then
      RUN="$(basename "$d")"; break
    fi
  done
  if [[ -z "$RUN" ]]; then
    echo "[train] resume: no run with a checkpoint found under $DIR/outputs/" >&2
    echo "        set RUN_NAME=<run> to choose one." >&2
    exit 1
  fi
  echo "[train] resume: auto-selected latest run '$RUN' (override with RUN_NAME=<run>)"
else
  RUN="smolvla_$(date +%Y%m%d_%H%M%S)"
fi

OUT="$DIR/outputs/$RUN"            # lerobot's output dir — it CREATES this and
                                   # refuses to start if it already exists.
LOGDIR="$DIR/logs/${RUN}"  # so logs + tb live BESIDE it, not inside.
LOG="$LOGDIR/train.log"

mkdir -p "$LOGDIR"

# TensorBoard: lerobot only logs to wandb, so tee stdout to a log and parse it
# into scalars in the background (tb_log.py). View: tensorboard --logdir "$LOGDIR/tb"
rm -f "$LOG.done"
"$VENV/bin/python" "$DIR/tb_log.py" "$LOG" "$LOGDIR/tb" &
TB_PID=$!
# On exit, tell the parser to finish (flush) and clean it up.
trap 'touch "$LOG.done"; wait "$TB_PID" 2>/dev/null || true' EXIT
echo "[train] TensorBoard scalars -> $LOGDIR/tb   (tensorboard --logdir $LOGDIR/tb)"

if [[ "$RESUME" == "1" ]]; then
  # Resume: lerobot loads the whole config from the checkpoint, so we pass only
  # --config_path + --resume (plus any explicit overrides like --steps). Append
  # to the existing log so the run's history stays in one file.
  CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
  if [[ ! -f "$CFG" ]]; then
    echo "[train] resume: no checkpoint config at $CFG" >&2
    echo "        set RUN_NAME=<existing run with a saved checkpoint>." >&2
    exit 1
  fi
  echo "[train] RESUMING $RUN from $(dirname "$CFG")"
  "$VENV/bin/lerobot-train" \
    --config_path="$CFG" \
    --resume=true \
    ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee -a "$LOG"
else
  # Use HuggingFace dataset repo (set HF_DATASET_REPO to override)
  HF_REPO="${HF_DATASET_REPO:-chanho-lee/lges_suction}"
  HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface/datasets}"

  "$VENV/bin/lerobot-train" \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id="$HF_REPO" \
    --dataset.root="$HF_CACHE_DIR" \
    --rename_map='{"observation.images.head": "observation.images.camera1", "observation.images.head_depth": "observation.images.camera2"}' \
    --policy.input_features='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}' \
    --batch_size=32 \
    --steps=60000 \
    --save_freq=2000 \
    --log_freq=50 \
    --num_workers=32 \
    --output_dir="$OUT" \
    --job_name="$RUN" \
    ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$LOG"
fi
