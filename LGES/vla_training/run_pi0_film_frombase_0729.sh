#!/usr/bin/env bash
# 0729 π0 FiLM round, frombase (2026-08-11). Two arms from lerobot/pi0_base that differ
# ONLY in the injection token — the token-level half of SmolVLA's prefix>suffix result
# (see film_contact_pi0.py header; π0.5 couldn't ask this, it has no state token):
#
#   GPU $GPU_STATE  : pi0_film_frombase_state_0729    (FILM_INJECT=state,  SmolVLA-'prefix' analogue)
#   GPU $GPU_ACTION : pi0_film_frombase_action_0729   (FILM_INJECT=action, 'suffix' analogue)
#
# Hyperparameters and FiLM calibration mirror run_pi05_0729_b300.sh exactly (batch 8,
# 50k steps, grad ckpt, cond=contact,fz,seal, mask_force=1, FZ_OFF=2.1 from
# logs/validate_0729.out FZ_MEDIAN) so pi0 cells sit next to the pi05 cells.
# onnaive arms are deliberately left for a later round ("일단 frombase로", 2026-08-11).
#
#   ./run_pi0_film_frombase_0729.sh                     # GPUs 4 + 6
#   GPU_STATE=5 GPU_ACTION=7 ./run_pi0_film_frombase_0729.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_STATE="${GPU_STATE:-4}"
GPU_ACTION="${GPU_ACTION:-6}"

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FILM_FZ_OFF=2.1   # from logs/validate_0729.out FZ_MEDIAN, same as the pi05 round

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[pi0_fb] datasets missing under $DIR/datasets" >&2; exit 1; }

# π0's preprocessor tokenizes with google/paligemma-3b-pt-224 (gated), same as π0.5.
"$PY" - <<'EOF' || exit 1
import sys
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
except Exception as e:
    print("[pi0_fb] cannot read google/paligemma-3b-pt-224 (pi0's tokenizer):",
          type(e).__name__, file=sys.stderr)
    print("[pi0_fb]   run: $VENV/bin/hf auth login   (or export HF_TOKEN=...)", file=sys.stderr)
    sys.exit(1)
print("[pi0_fb] paligemma tokenizer reachable")
EOF

mkdir -p "$DIR/logs"
FILM_ENV=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1
          FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT")

launch() {  # inject gpu
  local inject="$1" gpu="$2" run="pi0_film_frombase_${1}_0729"
  CUDA_VISIBLE_DEVICES="$gpu" env "${FILM_ENV[@]}" FILM_INJECT="$inject" \
    "$PY" "$DIR/train_film_pi0.py" \
    --policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
    --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
    --output_dir="$DIR/outputs/$run" --job_name="$run" \
    > "$DIR/logs/orch_${run}.out" 2>&1
}

launch state  "$GPU_STATE"  & P1=$!
launch action "$GPU_ACTION" & P2=$!
echo "[pi0_fb] pids state=$P1(GPU$GPU_STATE) action=$P2(GPU$GPU_ACTION)"

best() {  # run inject gpu
  local run="$1" inject="$2" gpu="$3"
  env CUDA_VISIBLE_DEVICES="$gpu" "${FILM_ENV[@]}" FILM_INJECT="$inject" \
    "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$run" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_pi0_0729.out" 2>&1 \
    && echo "[pi0_fb] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}

R1=0; wait $P1 || R1=$?; echo "[pi0_fb] state rc=$R1"
[[ $R1 == 0 ]] && best pi0_film_frombase_state_0729 state "$GPU_STATE"
R2=0; wait $P2 || R2=$?; echo "[pi0_fb] action rc=$R2"
[[ $R2 == 0 ]] && best pi0_film_frombase_action_0729 action "$GPU_ACTION"

echo "[pi0_fb] DONE state=$R1 action=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
