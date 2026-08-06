#!/usr/bin/env bash
# 0729 pi0.5 round, re-placed for the B300 box (2026-08-05). Same three runs and the same
# hyperparameters as run_pi05_0729.sh — naive, film-from-base, film_on_naive — only the GPU
# layout and the launch order differ, so the original stays as the record of the 0721/0727
# machine.
#
# Why a different layout: GPUs 5 and 7 (what run_pi05_0729.sh pins) are occupied by another
# user's multi-day jobs on this box, and only 4 and 6 are free. film-from-base starts from
# lerobot/pi05_base, so it does NOT depend on naive and is launched immediately instead of
# after it:
#
#   GPU $GPU_NAIVE : pi05_naive_0729
#   GPU $GPU_FILM  : pi05_film_frombase_0729   (starts now, independent of naive)
#                    pi05_film_onnaive_0729    (starts when naive's best ckpt exists)
#
# Each pi0.5 instance needs ~64 GB, so two co-resident on one 275 GB B300 is fine (two on a
# smaller card OOM'd once — that constraint is what the original's comment refers to).
# num_workers=32 as in the original (24+ needed to avoid a dataloader bottleneck).
#
#   ./run_pi05_0729_b300.sh                       # GPUs 4 + 6
#   GPU_NAIVE=0 GPU_FILM=1 ./run_pi05_0729_b300.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_NAIVE="${GPU_NAIVE:-4}"
GPU_FILM="${GPU_FILM:-6}"

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FILM_FZ_OFF=2.1   # from logs/validate_0729.out FZ_MEDIAN

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[pi05_b300] datasets missing under $DIR/datasets" >&2; exit 1; }

# pi0.5's preprocessor tokenizes with google/paligemma-3b-pt-224, a GATED repo: without a
# token that has accepted its license every run dies on a 401 well into startup. Fail here
# instead, with the fix.
"$PY" - <<'EOF' || exit 1
import sys
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
except Exception as e:
    print("[pi05_b300] cannot read google/paligemma-3b-pt-224 (pi0.5's tokenizer):",
          type(e).__name__, file=sys.stderr)
    print("[pi05_b300]   run: $VENV/bin/hf auth login   (or export HF_TOKEN=...)", file=sys.stderr)
    print("[pi05_b300]   the account must have accepted the license at", file=sys.stderr)
    print("[pi05_b300]   https://huggingface.co/google/paligemma-3b-pt-224", file=sys.stderr)
    sys.exit(1)
print("[pi05_b300] paligemma tokenizer reachable")
EOF

mkdir -p "$DIR/logs"
FILM_ENV=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1
          FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT")

# ---- launch naive (GPU_NAIVE) and film-from-base (GPU_FILM) together --------------
CUDA_VISIBLE_DEVICES="$GPU_NAIVE" "$PY" "$DIR/train_pi05.py" \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/pi05_naive_0729" --job_name=pi05_naive_0729 \
  > "$DIR/logs/orch_pi05_naive_0729.out" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES="$GPU_FILM" env "${FILM_ENV[@]}" "$PY" "$DIR/train_film_pi05.py" \
  --policy.path=lerobot/pi05_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/pi05_film_frombase_0729" --job_name=pi05_film_frombase_0729 \
  > "$DIR/logs/orch_pi05_film_frombase_0729.out" 2>&1 &
P2=$!
echo "[pi05_b300] pids naive=$P1(GPU$GPU_NAIVE) frombase=$P2(GPU$GPU_FILM)"

best() {  # run [film_env...]
  local run="$1"; shift
  env CUDA_VISIBLE_DEVICES="$GPU_FILM" "$@" "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$run" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_pi05_0729.out" 2>&1 \
    && echo "[pi05_b300] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}

# ---- film_on_naive waits for naive's best checkpoint ------------------------------
R1=0; wait $P1 || R1=$?; echo "[pi05_b300] naive rc=$R1"
if [[ $R1 != 0 ]]; then
  echo "[pi05_b300] naive failed -- skipping film_on_naive (it needs naive's checkpoint)"
else
  best pi05_naive_0729
  NAIVE_CKPT="$DIR/outputs/pi05_naive_0729/checkpoints/best/pretrained_model"
  CUDA_VISIBLE_DEVICES="$GPU_FILM" env "${FILM_ENV[@]}" "$PY" "$DIR/train_film_pi05.py" \
    --policy.path="$NAIVE_CKPT" --policy.device=cuda --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
    --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
    --output_dir="$DIR/outputs/pi05_film_onnaive_0729" --job_name=pi05_film_onnaive_0729 \
    > "$DIR/logs/orch_pi05_film_onnaive_0729.out" 2>&1 &
  P3=$!
  echo "[pi05_b300] pid onnaive=$P3(GPU$GPU_FILM)"
fi

R2=0; wait $P2 || R2=$?; echo "[pi05_b300] frombase rc=$R2"
[[ $R2 == 0 ]] && best pi05_film_frombase_0729 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"

R3=0
if [[ -n "${P3:-}" ]]; then
  wait "$P3" || R3=$?; echo "[pi05_b300] onnaive rc=$R3"
  [[ $R3 == 0 ]] && best pi05_film_onnaive_0729 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"
else
  R3=1
fi

echo "[pi05_b300] DONE naive=$R1 frombase=$R2 onnaive=$R3"
[[ $R1 == 0 && $R2 == 0 && $R3 == 0 ]]
