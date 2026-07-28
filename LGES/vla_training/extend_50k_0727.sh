#!/usr/bin/env bash
# Extend every 0721_0727 run to 50k steps and keep ONLY the final checkpoint
# (user request 2026-07-28). Completed runs resume immediately; still-running
# runs are resumed when their 30k finishes. All resumes rebuild the lr schedule
# to a 50k horizon (--scheduler.num_decay_steps) and use --save_freq=50000 so
# only the final checkpoint is written; intermediates are pruned afterwards.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
RT="$DIR/datasets/lges_case_pick_0721_0727"

wait_gone() {  # wait until no training process references this run's output_dir
  while pgrep -f "output_dir=.*$1" > /dev/null 2>&1; do sleep 120; done
}

prune() {  # keep only the checkpoint 'last' points to
  local ck="$DIR/outputs/$1/checkpoints"; [[ -d "$ck" ]] || return 0
  local keep; keep="$(readlink -f "$ck/last" 2>/dev/null | xargs -r basename)"
  for c in "$ck"/0*/; do
    [[ "$(basename "$c")" == "$keep" ]] || rm -rf "$c"
  done
  echo "[50k] pruned $1 -> kept $keep"
}

resume_smolvla() {  # run_name [extra env...]
  local run="$1"
  local cfg="$DIR/outputs/$run/checkpoints/last/pretrained_model/train_config.json"
  [[ -f "$cfg" ]] || { echo "[50k] no ckpt for $run" >&2; return 1; }
  "$PY" "$DIR/train_film.py" --config_path="$cfg" --resume=true \
    --steps=50000 --save_freq=10000 --scheduler.num_decay_steps=50000 \
    > "$DIR/logs/orch_${run}_50k.out" 2>&1
}

# ---- GPU 7: SmolVLA family -----------------------------------------------------
(
  export CUDA_VISIBLE_DEVICES=7
  # naive resumes through lerobot-train directly (no film env needed)
  "$PY" -m lerobot.scripts.lerobot_train \
    --config_path="$DIR/outputs/smolvla_naive_0721_0727/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --steps=50000 --save_freq=10000 --scheduler.num_decay_steps=50000 \
    > "$DIR/logs/orch_naive_0727_50k.out" 2>&1 &
  PN=$!
  FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
    FILM_DATASET_ROOT="$RT" resume_smolvla smolvla_film_0721_0727_prefix_mask1 &
  PF=$!
  FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
    FILM_OVERSAMPLE_BOOST=3 FILM_DATASET_ROOT="$RT" \
    resume_smolvla smolvla_film_0721_0727_prefix_mask1_os3 &
  PO=$!
  wait_gone smolvla_film_0721_0727_suffix_mask1
  FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_INJECT=suffix FILM_MASK_FORCE=1 \
    FILM_DATASET_ROOT="$RT" resume_smolvla smolvla_film_0721_0727_suffix_mask1 &
  PS=$!
  R=0; wait $PN || R=1; wait $PF || R=1; wait $PO || R=1; wait $PS || R=1
  for r in smolvla_naive_0721_0727 smolvla_film_0721_0727_prefix_mask1 \
           smolvla_film_0721_0727_prefix_mask1_os3 smolvla_film_0721_0727_suffix_mask1; do prune "$r"; done
  echo "[50k] GPU7 family done rc=$R"
) > "$DIR/logs/extend50k_gpu7.out" 2>&1 &
G7=$!

# ---- GPU 6: pi05 / xvla / act ---------------------------------------------------
(
  export CUDA_VISIBLE_DEVICES=6
  wait_gone pi05_naive_0721_0727
  "$PY" "$DIR/train_pi05.py" \
    --config_path="$DIR/outputs/pi05_naive_0721_0727/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --steps=50000 --save_freq=10000 --scheduler.num_decay_steps=50000 \
    > "$DIR/logs/orch_pi05_50k.out" 2>&1 &
  PP=$!
  wait_gone xvla_0721_0727
  "$PY" "$DIR/train_xvla.py" \
    --config_path="$DIR/outputs/xvla_0721_0727/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true --steps=50000 --save_freq=10000 --scheduler.num_decay_steps=50000 \
    > "$DIR/logs/orch_xvla_50k.out" 2>&1 &
  PX=$!
  wait_gone act_0721_0727        # ACT already targets 50k; just wait then prune
  R=0; wait $PP || R=1; wait $PX || R=1
  for r in pi05_naive_0721_0727 xvla_0721_0727 act_0721_0727; do prune "$r"; done
  echo "[50k] GPU6 family done rc=$R"
) > "$DIR/logs/extend50k_gpu6.out" 2>&1 &
G6=$!

RA=0; wait $G7 || RA=1; wait $G6 || RA=1
cat "$DIR/logs/extend50k_gpu7.out" "$DIR/logs/extend50k_gpu6.out" | grep "\[50k\]"
echo "[50k] ALL DONE rc=$RA"; exit $RA
