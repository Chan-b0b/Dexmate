#!/usr/bin/env bash
# 0729 π0 naive arm (2026-08-11): the no-FiLM baseline for the pi0 frombase pair
# (run_pi0_film_frombase_0729.sh) — required by the offline probe battery, and later
# usable as the warm start for an onnaive round. Mirrors pi05_naive_0729's
# hyperparameters exactly. train_pi05.py is policy-agnostic (it only registers the
# relative_actions_processor shim, which pi0_base's preprocessor also needs), so it
# launches π0 unchanged.
#
#   ./run_pi0_naive_0729.sh          # GPU 5
#   GPU=7 ./run_pi0_naive_0729.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU="${GPU:-5}"
REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
RUN=pi0_naive_0729

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[pi0_naive] datasets missing under $DIR/datasets" >&2; exit 1; }
mkdir -p "$DIR/logs"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$DIR/train_pi05.py" \
  --policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/$RUN" --job_name="$RUN" \
  > "$DIR/logs/orch_${RUN}.out" 2>&1
R=$?
echo "[pi0_naive] train rc=$R"
if [[ $R == 0 ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$RUN" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_pi0_0729.out" 2>&1 \
    && echo "[pi0_naive] best: $(readlink "$DIR/outputs/$RUN/checkpoints/best" 2>/dev/null)"
fi
echo "[pi0_naive] DONE rc=$R"
exit $R
