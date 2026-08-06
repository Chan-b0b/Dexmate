#!/usr/bin/env bash
# pi0.5 FiLM offline probe battery — SERVER version (local checkpoints, no HF).
# Covers the two FiLM arms the naive battery (probe_0729_pi05_server.sh) leaves out:
#   fb = pi05_film_frombase_0729   (FiLM from lerobot/pi05_base)
#   on = pi05_film_onnaive_0729    (FiLM warm-started from pi05_naive_0729 best@10k)
#
# HOW TO READ THIS (scope limit — do not skip):
#   pi0.5-FiLM is SUFFIX-ONLY (no state token, so SmolVLA's winning 'prefix' point has no
#   analogue) and was trained with the OLD calibration: 3 channels (contact,fz,seal) and
#   FZ_OFF=2.1. The SmolVLA recal runs are prefix + 4 channels (contact,fmag,fz,seal) with
#   F0=5.5/tau=1/fz=(fz-3.0)/0.7. The numbers are therefore NOT directly comparable.
#   Use this battery only for "does the sign/shape of the effect reproduce on another
#   architecture", never for magnitude comparisons against the SmolVLA cells.
#
# The FiLM env below MUST match training (run_pi05_0729_b300.sh): cond=contact,fz,seal,
# mask_force=1, FZ_OFF=2.1; F0/TAU/FZ_TAU stay at the film_contact defaults 6/4/5, which is
# what training used. Getting these wrong silently re-calibrates c-hat and invalidates every
# cell.
#
# Requires the pi05 probe routing added 2026-08-06: probe_state_authority/probe_press_sim/
# eval_offline take --film-pi05, which patches via film_contact_pi05 (quantile-normalized
# state) AND installs the forced-c hook on film_contact_pi05._cond_from_state.
#
# Usage: ./probe_0729_pi05film_server.sh              (GPU=<n>, default 7)
#        GPU=7 RUNS=fb ./probe_0729_pi05film_server.sh    # one arm per GPU, in parallel
#        GPU=6 RUNS=on CKPTS=best ./probe_0729_pi05film_server.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
RV="$DIR/datasets/lges_case_pick_0729_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val

# training-matched FiLM calibration (see header)
export FILM_COND=contact,fz,seal
export FILM_MASK_FORCE=1
export FILM_FZ_OFF=2.1

[[ -d "$RV/meta" ]] || { echo "[pi05film] val dataset missing: $RV" >&2; exit 1; }
mkdir -p "$DIR/probes"

for arm in ${RUNS:-fb on}; do
  case "$arm" in
    fb) RUN="$DIR/outputs/pi05_film_frombase_0729" ;;
    on) RUN="$DIR/outputs/pi05_film_onnaive_0729" ;;
    *)  echo "[pi05film] unknown arm '$arm' (want fb|on)" >&2; exit 1 ;;
  esac
  [[ -d "$RUN/checkpoints" ]] || { echo "[pi05film] skip $arm (no checkpoints)"; continue; }

  for ck in ${CKPTS:-best last}; do
    CK="$RUN/checkpoints/$ck"
    [[ -e "$CK" ]] || { echo "[pi05film] skip $arm/$ck (missing)"; continue; }
    TAG="pi05film${arm}"
    state() { # <outfile> <extra...>
      local out=$1; shift
      "$PY" "$DIR/probe_state_authority.py" --film-pi05 --all-episodes \
        --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" "$@" \
        > "$DIR/probes/$out" 2>&1
      echo "[pi05film] $out rc=$? -> $(grep -a -m1 'ALL frames' "$DIR/probes/$out" | sed 's/^ *//')"
    }
    state "0729_state_${TAG}_${ck}_pc_fc.txt"  --swap firstcontact --pre-contact 10
    state "0729_state_${TAG}_${ck}_ramp8.txt"  --swap fcscale --fc-mag 8
    state "0729_state_${TAG}_${ck}_ramp12.txt" --swap fcscale --fc-mag 12
    state "0729_state_${TAG}_${ck}_pc_r12.txt" --swap fcscale --fc-mag 12 --pre-contact 10
    for off in 1 30; do
      "$PY" "$DIR/probe_press_sim.py" --film-pi05 --checkpoint "$CK" \
        --dataset-root "$RV" --repo-id "$REPO_VAL" \
        --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
        > "$DIR/probes/0729_sim_${TAG}_${ck}_off${off}_fzdelta.txt" 2>&1
      echo "[pi05film] sim $arm/$ck off$off rc=$?"
    done
    "$PY" "$DIR/eval_offline.py" --film-pi05 --checkpoint "$CK" \
      --val-root "$RV" --repo-id "$REPO_VAL" \
      > "$DIR/probes/0729_eval_${TAG}_${ck}.txt" 2>&1
    echo "[pi05film] eval $arm/$ck rc=$?"
  done
done

echo; echo "===== pi05-FiLM verdict material (sign/shape only — see header) ====="
grep -a -H "ALL frames" "$DIR"/probes/0729_state_pi05film*_*.txt 2>/dev/null
grep -a -H -A3 "── summary" "$DIR"/probes/0729_sim_pi05film*_*.txt 2>/dev/null | grep -v "^--$"
grep -a -H "OVERALL" "$DIR"/probes/0729_eval_pi05film*_*.txt 2>/dev/null
echo "[pi05film] DONE — commit probes/ + push"
