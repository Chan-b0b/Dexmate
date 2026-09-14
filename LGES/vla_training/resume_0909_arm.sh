#!/usr/bin/env bash
# Resume ONE 0909 arm on a different GPU, then finish its chain (2026-09-09).
# Needed because the user had to free GPUs 0-1 mid-round: killing an arm makes its
# chain() in run_case_pick_0909_all.sh exit rc!=0 and SKIP select_best + battery, so
# the tail of the pipeline has to be re-driven here.
#
# Resume semantics (lerobot 0.5.1 configs/train.py:92): pass ONLY --config_path +
# --resume=true. Passing --policy.path instead takes the from-pretrained branch and
# would silently restart from step 0. The checkpoint's training_state/ carries
# optimizer + scheduler + rng, so the run continues exactly where it stopped.
# The FiLM patch is applied at import time by train_film_pi0.py, so it survives resume.
#
# Usage: ./resume_0909_arm.sh <run_name> <gpu> <kind> <inject|-> <entry.py>
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

NAME=${1:?run name}; GPU=${2:?gpu}; KIND=${3:?kind}; INJECT=${4:?inject}; ENTRY=${5:?entry.py}
RT="$DIR/datasets/lges_case_pick_0909"
RV="$DIR/datasets/lges_case_pick_0909_val"
REPO_VAL_ID=Chanho-Lee/lges_case_pick_0909_val
V="$DIR/logs/validate_0909.out"
cal() { grep -o "^CAL_$1=.*" "$V" | tail -1 | cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZ="$(cal FZ_OFF)"; FZT="$(cal FZ_TAU)"; C1="$(cal C1)"
FZD="$(cal FZ_DELTA0)"; FBASE="$(cal F_BASE)"; FCAP="$(cal F_CAP)"; PRECON="$(cal PRE_CONTACT)"; FCFZ="$(cal FC_FZ)"
[[ -n "$F0" && -n "$TAU" && -n "$FZ" && -n "$FZT" && -n "$C1" ]] \
  || { echo "[$NAME] calibration missing from $V" >&2; exit 1; }
FB=(FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_MASK_FORCE=1
    FILM_F0="$F0" FILM_TAU="$TAU" FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZ"
    FILM_DATASET_ROOT="$RT")

CFG="$DIR/outputs/$NAME/checkpoints/last/pretrained_model/train_config.json"
[[ -f "$CFG" ]] || { echo "[$NAME] no checkpoint config at $CFG" >&2; exit 1; }
echo "[$NAME] resuming on GPU $GPU from $(readlink "$DIR/outputs/$NAME/checkpoints/last")"

# ENTRY is either a script in $DIR or "-m <module>" (smolvla_naive has no FiLM launcher
# and trains through lerobot's own entry point).
if [[ "$ENTRY" == -m* ]]; then RUNPY=(-m "${ENTRY#-m }"); else RUNPY=("$DIR/$ENTRY"); fi
CUDA_VISIBLE_DEVICES="$GPU" env "${FB[@]}" FILM_INJECT="${INJECT/-/state}" \
  "$PY" "${RUNPY[@]}" --config_path="$CFG" --resume=true \
  >> "$DIR/logs/orch_${NAME}.out" 2>&1
rc=$?
echo "[$NAME] train rc=$rc"
[[ $rc == 0 ]] || exit $rc

env CUDA_VISIBLE_DEVICES="$GPU" "${FB[@]}" FILM_INJECT="${INJECT/-/state}" \
  "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$NAME" \
  --val-root "$RV" --repo-id "$REPO_VAL_ID" --prune >> "$DIR/logs/best_0909.out" 2>&1
echo "[$NAME] best: $(readlink "$DIR/outputs/$NAME/checkpoints/best" 2>/dev/null)"

VAL_ROOT="$RV" REPO_VAL="$REPO_VAL_ID" FZ_OFF="$FZ" FZ_TAU="$FZT" F0="$F0" TAU="$TAU" C1="$C1" \
    FZ_DELTA0="$FZD" F_BASE="$FBASE" F_CAP="$FCAP" PRE_CONTACT="$PRECON" FC_FZ="$FCFZ" PFX=0909 \
  "$DIR/run_battery_0729cal.sh" "$NAME" "$KIND" "$INJECT" \
  "$DIR/outputs/$NAME/checkpoints/best" "$GPU" >> "$DIR/logs/battery_0909.out" 2>&1
echo "[$NAME] battery rc=$?"
