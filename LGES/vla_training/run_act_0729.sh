#!/usr/bin/env bash
# 0729 ACT round, part 1 (2026-08-14): naive + FiLM-state in parallel — the third
# architecture for the token-level injection story (SmolVLA prefix / π0 state / ACT
# encoder state-token, film_contact_act.py).
#
#   GPU $GPU_NAIVE : act_0729               (naive ACT from scratch — ACT has no pretrained
#                                            base, so naive IS the from-scratch baseline)
#   GPU $GPU_FILM  : act_film_scratch_0729  (state-token FiLM learned jointly from scratch —
#                                            the 'frombase state' analogue)
#
# Hyperparameters follow run_act_film_0729.sh (bs 32, 50k steps, save 5k, nw 16); FiLM
# calibration matches the pi0/pi05 rounds (cond=contact,fz,seal mask_force=1 FZ_OFF=2.1,
# F0/TAU/FZ_TAU defaults 6/4/5). No rename_map: ACT trains from scratch on the dataset's
# own camera keys. act_film_onnaive_0729 stays in run_act_film_0729.sh (needs act_0729 best).
#
#   ./run_act_0729.sh                       # GPUs 4 + 5
#   GPU_NAIVE=0 GPU_FILM=1 ./run_act_0729.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_NAIVE="${GPU_NAIVE:-4}"
GPU_FILM="${GPU_FILM:-5}"

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
FILM_FZ_OFF=2.1   # from logs/validate_0729.out FZ_MEDIAN, same as pi0/pi05 rounds

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[act_0729] datasets missing under $DIR/datasets" >&2; exit 1; }
mkdir -p "$DIR/logs"

CUDA_VISIBLE_DEVICES="$GPU_NAIVE" "$PY" -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=32 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/act_0729" --job_name=act_0729 \
  > "$DIR/logs/orch_act_0729.out" 2>&1 &
P1=$!

CUDA_VISIBLE_DEVICES="$GPU_FILM" env FILM_VARIANT=v2 FILM_COND=contact,fz,seal \
  FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_act.py" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=32 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/act_film_scratch_0729" --job_name=act_film_scratch_0729 \
  > "$DIR/logs/orch_act_film_scratch_0729.out" 2>&1 &
P2=$!
echo "[act_0729] pids naive=$P1(GPU$GPU_NAIVE) film=$P2(GPU$GPU_FILM)"

best() { # run gpu film_env...
  local run="$1" gpu="$2"; shift 2
  env CUDA_VISIBLE_DEVICES="$gpu" "$@" "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$run" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_act_0729.out" 2>&1 \
    && echo "[act_0729] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}

R1=0; wait $P1 || R1=$?; echo "[act_0729] naive rc=$R1"
[[ $R1 == 0 ]] && best act_0729 "$GPU_NAIVE"
R2=0; wait $P2 || R2=$?; echo "[act_0729] film rc=$R2"
[[ $R2 == 0 ]] && best act_film_scratch_0729 "$GPU_FILM" \
  FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"

echo "[act_0729] DONE naive=$R1 film=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
