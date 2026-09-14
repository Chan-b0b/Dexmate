#!/usr/bin/env bash
# 0729-round 'layers' arms (2026-08-14): per-layer FiLM on the action decoder/expert
# MLP branch (inject='layers', film_contact.py / film_contact_act.py) — the
# architecture-GENERIC injection point, to compare against the token-level arms:
#   ACT     : act_0729 (naive) / act_film_scratch_0729 (state token)   [GPUs 4/5]
#   SmolVLA : smolvla_naive_0729 / smolvla_film_0729_prefix_mask1 / _suffix_mask1
# Arms here:
#   GPU $GPU_ACT     : act_film_layers_0729     (from scratch — act_0729 recipe, bs32 50k)
#   GPU $GPU_SMOLVLA : smolvla_film_layers_0729 (from smolvla_base — 0729 film recipe, 50k)
# After each: select_best_ckpt (val loss, --prune). SmolVLA also gets the offline
# counterfactual probe (probe_film_authority, std + realistic c patterns) on VAL.
#
#   ./run_film_layers_0814.sh                  # GPUs 6 + 4
#   GPU_ACT=7 GPU_SMOLVLA=5 ./run_film_layers_0814.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

GPU_ACT="${GPU_ACT:-6}"
GPU_SMOLVLA="${GPU_SMOLVLA:-4}"

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
FILM_FZ_OFF=2.1   # logs FZ_MEDIAN, same as the act/pi0/pi05 0729 rounds
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
FENV=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_INJECT=layers
      FILM_FZ_OFF="$FILM_FZ_OFF")

[[ -d "$RT/meta" && -d "$RV/meta" ]] || { echo "[layers_0814] datasets missing under $DIR/datasets" >&2; exit 1; }
mkdir -p "$DIR/logs" "$DIR/probes"

CUDA_VISIBLE_DEVICES="$GPU_ACT" env "${FENV[@]}" FILM_DATASET_ROOT="$RT" \
  "$PY" "$DIR/train_film_act.py" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id="$REPO" --dataset.root="$RT" \
  --batch_size=32 --steps=50000 --save_freq=5000 --log_freq=100 --num_workers=16 \
  --output_dir="$DIR/outputs/act_film_layers_0729" --job_name=act_film_layers_0729 \
  > "$DIR/logs/orch_act_film_layers_0729.out" 2>&1 &
P1=$!

BASE="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"
CUDA_VISIBLE_DEVICES="$GPU_SMOLVLA" env "${FENV[@]}" \
  RUN_NAME=smolvla_film_layers_0729 INIT_CKPT="$BASE" \
  DATASET_REPO="$REPO" DATASET_ROOT="$RT" FILM_DATASET_ROOT="$RT" \
  "$DIR/train_film.sh" --policy.input_features="$IF15" --steps=50000 --save_freq=5000 \
  --policy.scheduler_decay_steps=50000 \
  > "$DIR/logs/orch_smolvla_film_layers_0729.out" 2>&1 &
P2=$!
echo "[layers_0814] pids act=$P1(GPU$GPU_ACT) smolvla=$P2(GPU$GPU_SMOLVLA)"

best() { # run gpu
  local run="$1" gpu="$2"
  env CUDA_VISIBLE_DEVICES="$gpu" "${FENV[@]}" "$PY" "$DIR/select_best_ckpt.py" \
    --run "$DIR/outputs/$run" --val-root "$RV" --repo-id "$REPO_VAL" --prune \
    >> "$DIR/logs/best_layers_0814.out" 2>&1 \
    && echo "[layers_0814] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}
probe() { # tag [c0 c1]
  env CUDA_VISIBLE_DEVICES="$GPU_SMOLVLA" "${FENV[@]}" \
    "$PY" "$DIR/probe_film_authority.py" \
    --checkpoint "$DIR/outputs/smolvla_film_layers_0729/checkpoints/best" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
    ${2:+--c0 "$2" --c1 "$3"} > "$DIR/probes/0729_layers_$1.txt" 2>&1
  echo "[layers_0814] probe $1 rc=$?"
}

R1=0; wait $P1 || R1=$?; echo "[layers_0814] act rc=$R1"
[[ $R1 == 0 ]] && best act_film_layers_0729 "$GPU_ACT"
R2=0; wait $P2 || R2=$?; echo "[layers_0814] smolvla rc=$R2"
if [[ $R2 == 0 ]]; then
  best smolvla_film_layers_0729 "$GPU_SMOLVLA"
  probe std
  probe real "0,0,0" "0.6,0.42,0"
fi
echo "[layers_0814] DONE act=$R1 smolvla=$R2"
[[ $R1 == 0 && $R2 == 0 ]]
