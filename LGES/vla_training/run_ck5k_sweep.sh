#!/usr/bin/env bash
# Per-checkpoint probe sweep for the _r2 runs (2026-08-20): for EVERY 5k checkpoint of
# one arm, run the dose-response cells (pc_fc + ramp 8/10/12/15/20) and the press sim
# (off1/off30) — the authority-vs-training curve the r2 retrain exists for.
# Usage: ./run_ck5k_sweep.sh <run_name> <kind> <inject> <gpu>
#   e.g. ./run_ck5k_sweep.sh pi0_film_layers_0816_r2 film-pi0 layers 2
# Outputs: probes/ck5k_<run>_<step>_{pc_fc,ramp*,sim_off*}.txt
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
export HF_HOME="$HOME/.cache/huggingface"

NAME=${1:?run name}; KIND=${2:?naive|film-pi0|film-groot}; INJECT=${3:?inject}; GPU=${4:?gpu}
RV="$DIR/datasets/lges_case_pick_0816_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0816_val
FENV=(FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5
      FILM_FZ_OFF=1.4 FILM_INJECT="$INJECT" CUDA_VISIBLE_DEVICES="$GPU")
FLAG=()
case "$KIND" in
  naive)      FLAG=(--naive) ;;
  film-pi0)   FLAG=(--film-pi0) ;;
  film-groot) FLAG=(--film-groot) ;;
  *) echo "unknown kind $KIND" >&2; exit 1 ;;
esac

mkdir -p "$DIR/probes"
for CK in "$DIR/outputs/$NAME/checkpoints"/0*; do
  [[ -d "$CK/pretrained_model" ]] || continue
  STEP=$(basename "$CK")
  for mag in 8 10 12 15 20; do
    env "${FENV[@]}" "$PY" "$DIR/probe_state_authority.py" "${FLAG[@]}" \
      --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" \
      --all-episodes --swap fcscale --fc-mag "$mag" \
      > "$DIR/probes/ck5k_${NAME}_${STEP}_ramp${mag}.txt" 2>&1
    echo "[$NAME/$STEP] ramp$mag rc=$?"
  done
  env "${FENV[@]}" "$PY" "$DIR/probe_state_authority.py" "${FLAG[@]}" \
    --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" \
    --all-episodes --swap firstcontact --pre-contact 10 \
    > "$DIR/probes/ck5k_${NAME}_${STEP}_pc_fc.txt" 2>&1
  echo "[$NAME/$STEP] pc_fc rc=$?"
  for off in 1 30; do
    env "${FENV[@]}" "$PY" "$DIR/probe_press_sim.py" "${FLAG[@]}" \
      --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" \
      --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
      > "$DIR/probes/ck5k_${NAME}_${STEP}_sim_off${off}.txt" 2>&1
    echo "[$NAME/$STEP] sim off$off rc=$?"
  done
done
echo "[$NAME] SWEEP DONE"
