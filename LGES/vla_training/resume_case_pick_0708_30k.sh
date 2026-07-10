#!/usr/bin/env bash
# Extend the four 0708 case_pick runs from 10k to 30k steps (losses were still
# falling at 10k; the smolvla cosine schedule is designed for 30k and was being
# auto-compressed to fit 10k — resuming with --steps=30000 rebuilds the full
# 30k schedule, warm-restarting lr at ~7.5e-5 from step 10k).
# All four resume in parallel on GPU 6; each film_on_naive run (2k steps) starts
# from its naive run's 30k checkpoint as soon as that naive run finishes.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=6
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

ROOT_D="$DIR/datasets/lges_case_pick_0708"
ROOT_A="$DIR/datasets/lges_case_pick_0708_abs"
REPO_D=Chanho-Lee/lges_case_pick_0708
REPO_A=Chanho-Lee/lges_case_pick_0708_abs
mkdir -p "$DIR/logs"

resume_film() {  # run_name film_dataset_root
  local run="$1" root="$2"
  local cfg="$DIR/outputs/$run/checkpoints/last/pretrained_model/train_config.json"
  [[ -f "$cfg" ]] || { echo "[orch-30k] missing $cfg" >&2; return 1; }
  FILM_VARIANT=v2 FILM_DATASET_ROOT="$root" "$PY" "$DIR/train_film.py" \
    --config_path="$cfg" --resume=true --steps=30000
}

echo "[orch-30k] resuming 4 runs to 30k steps in parallel on GPU 6"

RUN_NAME=smolvla_naive_0708 "$DIR/train_smolvla.sh" --resume --steps=30000 \
  > "$DIR/logs/orch_naive_30k.out" 2>&1 &
P_ND=$!
RUN_NAME=smolvla_naive_0708_abs "$DIR/train_smolvla.sh" --resume --steps=30000 \
  > "$DIR/logs/orch_naive_abs_30k.out" 2>&1 &
P_NA=$!
resume_film smolvla_film_0708 "$ROOT_D" \
  > "$DIR/logs/orch_film_base_30k.out" 2>&1 &
P_FD=$!
resume_film smolvla_film_0708_abs "$ROOT_A" \
  > "$DIR/logs/orch_film_base_abs_30k.out" 2>&1 &
P_FA=$!

echo "[orch-30k] pids naive=$P_ND naive_abs=$P_NA film=$P_FD film_abs=$P_FA"

RC_ND=0; wait "$P_ND" || RC_ND=$?
echo "[orch-30k] naive (delta) finished rc=$RC_ND"
RC_FN=-1
if [[ "$RC_ND" == "0" ]]; then
  echo "[orch-30k] launching film_on_naive_0708 (2k) from 30k naive checkpoint"
  FILM_VARIANT=v2 RUN_NAME=film_on_naive_0708 \
    INIT_CKPT="$DIR/outputs/smolvla_naive_0708/checkpoints/last/pretrained_model" \
    DATASET_REPO="$REPO_D" DATASET_ROOT="$ROOT_D" FILM_DATASET_ROOT="$ROOT_D" \
    "$DIR/train_film.sh" --steps=2000 \
    > "$DIR/logs/orch_film_naive.out" 2>&1 &
  P_FND=$!
else
  echo "[orch-30k] film_on_naive_0708 SKIPPED (naive rc=$RC_ND)" >&2
fi

RC_NA=0; wait "$P_NA" || RC_NA=$?
echo "[orch-30k] naive (abs) finished rc=$RC_NA"
RC_FNA=-1
if [[ "$RC_NA" == "0" ]]; then
  echo "[orch-30k] launching film_on_naive_0708_abs (2k) from 30k naive checkpoint"
  FILM_VARIANT=v2 RUN_NAME=film_on_naive_0708_abs \
    INIT_CKPT="$DIR/outputs/smolvla_naive_0708_abs/checkpoints/last/pretrained_model" \
    DATASET_REPO="$REPO_A" DATASET_ROOT="$ROOT_A" FILM_DATASET_ROOT="$ROOT_A" \
    "$DIR/train_film.sh" --steps=2000 \
    > "$DIR/logs/orch_film_naive_abs.out" 2>&1 &
  P_FNA=$!
else
  echo "[orch-30k] film_on_naive_0708_abs SKIPPED (naive rc=$RC_NA)" >&2
fi

RC_FD=0; wait "$P_FD" || RC_FD=$?
echo "[orch-30k] film_from_base (delta) finished rc=$RC_FD"
RC_FA=0; wait "$P_FA" || RC_FA=$?
echo "[orch-30k] film_from_base (abs) finished rc=$RC_FA"
[[ "${P_FND:-}" ]] && { RC_FN=0;  wait "$P_FND" || RC_FN=$?;  echo "[orch-30k] film_on_naive (delta) finished rc=$RC_FN"; }
[[ "${P_FNA:-}" ]] && { RC_FNA=0; wait "$P_FNA" || RC_FNA=$?; echo "[orch-30k] film_on_naive (abs) finished rc=$RC_FNA"; }

echo "[orch-30k] DONE naive=$RC_ND naive_abs=$RC_NA film=$RC_FD film_abs=$RC_FA film_on_naive=$RC_FN film_on_naive_abs=$RC_FNA"
[[ "$RC_ND" == 0 && "$RC_NA" == 0 && "$RC_FD" == 0 && "$RC_FA" == 0 && "$RC_FN" == 0 && "$RC_FNA" == 0 ]]
