#!/usr/bin/env bash
# 0729 FROMNAIVE-MASK0 round (2026-08-04): NEGATIVE CONTROL — same as fromnaive but
# warm-started from the TRAINED naive checkpoint (smolvla_naive_0729 best@10k) instead of
# smolvla_base. Same structure as pm1_recal, different init — the task skills are already
# learned, so the fine-tune only has to re-route force knowledge through c-hat (mask1
# removes the raw wrench the naive weights used; expect a short adaptation transient).
# 20k steps, save every 2500 for a dense val curve. GPU 7.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
# GPU 7 by default (the 0721/0727 box); override with GPU=<n> — on the B300 box 7 holds
# another user's job, so this round runs on 4 once the v1 control releases it.
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'

RECAL_ENV=(FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix
           FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1
           FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7)
RUN=smolvla_film_0729_prefix_mask0_recal_fromnaive
INIT="$DIR/outputs/smolvla_naive_0729/checkpoints/best/pretrained_model"

[[ -d "$RT/meta" && -d "$RV/meta" && -d "$INIT" ]] || { echo "[fromnaive_m0] inputs missing" >&2; exit 1; }
mkdir -p "$DIR/logs" "$DIR/probes"

env "${RECAL_ENV[@]}" FILM_VARIANT=v2 FILM_MASK_FORCE=0 \
  RUN_NAME="$RUN" INIT_CKPT="$INIT" \
  DATASET_REPO="$REPO" DATASET_ROOT="$RT" FILM_DATASET_ROOT="$RT" \
  "$DIR/train_film.sh" --policy.input_features="$IF15" --steps=20000 --save_freq=2500 \
  --policy.scheduler_decay_steps=20000 \
  > "$DIR/logs/orch_film_0729_fromnaive_m0.out" 2>&1
R=$?; echo "[fromnaive_m0] train rc=$R"
[[ $R == 0 ]] || exit $R

env "${RECAL_ENV[@]}" FILM_MASK_FORCE=0 \
  "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$RUN" \
  --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_0729_fromnaive_m0.out" 2>&1
echo "[fromnaive_m0] best: $(readlink "$DIR/outputs/$RUN/checkpoints/best" 2>/dev/null)"

for ck in best last; do
  for off in 1 30; do
    for fm in pattern fzdelta; do
      env "${RECAL_ENV[@]}" FILM_MASK_FORCE=0 \
        "$PY" "$DIR/probe_press_sim.py" --checkpoint "$DIR/outputs/$RUN/checkpoints/$ck" \
        --dataset-root "$RV" --repo-id "$REPO_VAL" \
        --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model "$fm" \
        > "$DIR/probes/0729_sim_fromnaive_m0_${ck}_off${off}_${fm}.txt" 2>&1
      echo "[fromnaive_m0] sim $ck off$off $fm rc=$?"
    done
  done
  env "${RECAL_ENV[@]}" FILM_MASK_FORCE=0 \
    "$PY" "$DIR/probe_state_authority.py" --swap firstcontact --all-episodes --pre-contact 10 \
    --checkpoint "$DIR/outputs/$RUN/checkpoints/$ck" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" \
    > "$DIR/probes/0729_state_fromnaive_m0_${ck}_pc_fc.txt" 2>&1
  echo "[fromnaive_m0] pc_fc $ck rc=$?"
  env "${RECAL_ENV[@]}" FILM_MASK_FORCE=0 \
    "$PY" "$DIR/probe_state_authority.py" --swap fcscale --fc-mag 12 --all-episodes --pre-contact 10 \
    --checkpoint "$DIR/outputs/$RUN/checkpoints/$ck" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" \
    > "$DIR/probes/0729_state_fromnaive_m0_${ck}_pc_r12.txt" 2>&1
  echo "[fromnaive_m0] pc_r12 $ck rc=$?"
done
grep -a -H -A4 "── summary" "$DIR"/probes/0729_sim_fromnaive_m0_*.txt | grep -v "^--$"
echo "[fromnaive_m0] DONE"
