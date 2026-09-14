#!/usr/bin/env bash
# 0729-round 'layers' arms, part 2 (2026-08-16): π0 + GR00T per-layer FiLM
# (inject='layers', film_contact_pi0.py / film_contact_groot.py) — completes the
# architecture-generic injection matrix:
#   SmolVLA layers (done, best@10k) / ACT layers (done, best@35k) / π0 / GR00T
# Arms here (recipes = run_pi0_film_frombase_0729.sh / run_groot_0729.sh, only
# FILM_INJECT differs):
#   GPU $GPU_PI0   : pi0_film_frombase_layers_0729  (from lerobot/pi0_base, bs8 50k)
#   GPU $GPU_GROOT : groot_film_layers_0729         (--policy.type=groot,   bs8 50k)
# After each: select_best_ckpt (val loss, --prune, FILM_INJECT=layers).
#
#   ./run_film_layers2_0816.sh                 # GPUs 4 + 5
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_PI0="${GPU_PI0:-4}"
GPU_GROOT="${GPU_GROOT:-5}"

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
RENAME='{"observation.images.head": "observation.images.base_0_rgb", "observation.images.head_depth": "observation.images.left_wrist_0_rgb"}'
FILM_FZ_OFF=2.1
FENV=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_INJECT=layers
      FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT")

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[layers2] datasets missing under $DIR/datasets" >&2; exit 1; }
mkdir -p "$DIR/logs"

CUDA_VISIBLE_DEVICES="$GPU_PI0" env "${FENV[@]}" \
  "$PY" "$DIR/train_film_pi0.py" \
  --policy.path=lerobot/pi0_base --policy.device=cuda --policy.push_to_hub=false \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id="$REPO" --dataset.root="$RT" --rename_map="$RENAME" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=32 \
  --output_dir="$DIR/outputs/pi0_film_frombase_layers_0729" \
  --job_name=pi0_film_frombase_layers_0729 \
  > "$DIR/logs/orch_pi0_film_frombase_layers_0729.out" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES="$GPU_GROOT" env "${FENV[@]}" \
  "$PY" "$DIR/train_film_groot.py" \
  --policy.type=groot --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=8 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/groot_film_layers_0729" --job_name=groot_film_layers_0729 \
  > "$DIR/logs/orch_groot_film_layers_0729.out" 2>&1 &
P2=$!
echo "[layers2] pids pi0=$P1(GPU$GPU_PI0) groot=$P2(GPU$GPU_GROOT)"

best() { # run gpu
  local run="$1" gpu="$2"
  env CUDA_VISIBLE_DEVICES="$gpu" "${FENV[@]}" "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$run" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_layers2_0816.out" 2>&1 \
    && echo "[layers2] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}

R1=0; wait $P1 || R1=$?; echo "[layers2] pi0 rc=$R1"
[[ $R1 == 0 ]] && best pi0_film_frombase_layers_0729 "$GPU_PI0"
R2=0; wait $P2 || R2=$?; echo "[layers2] groot rc=$R2"
[[ $R2 == 0 ]] && best groot_film_layers_0729 "$GPU_GROOT"
echo "[layers2] DONE pi0=$R1 groot=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
