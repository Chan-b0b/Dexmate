#!/usr/bin/env bash
# Robot eval — 0729 recal round (fromnaive vs naive, optional pm1_recal A/B).
# Offline evidence (probes/0729_*): fromnaive = state-swap 94-97% descent cancel
# (all via FiLM path), press-sim 6/6 stop w/ lowest penetration, val action err equal
# to naive -> new #1 candidate. Key on-robot test: middle layer (L3) not in training.
#
# Usage: ./robot_eval_0729_recal.sh <target> <layer> [extra run_policy args...]
#   target: baseline | naive | fromnaive_best | fromnaive_last | pm1r_best | pm1r_last | live_probe
#   layer : L1 | L3 | L5  (log subdir; baseline/live_probe ignore it)
# Prereqs:
#   1. checkpoints on HF (fromnaive + v1 uploaded; pm1r still needs
#      upload_0729_recal_server.sh on the SERVER)
#   2. FiLM offsets self-anchor against the live wrench in both paths:
#      eval runs at task start (--film-auto-baseline), live probe at the
#      pre-descent hover (--baseline-hover). Values are logged (meta.json /
#      probe JSON). The `baseline` target is a manual diagnostic only.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

TARGET=${1:?usage: $0 <target> <layer>}
if [[ "$TARGET" == live_probe || "$TARGET" == baseline ]]; then
  LAYER=""   # no layer arg — everything after <target> passes through
  shift 1
else
  LAYER=${2:?layer required (L1|L3|L5)}
  [[ "$LAYER" =~ ^L[0-9]+$ ]] || { echo "layer must be L<n>, got '$LAYER'" >&2; exit 1; }
  shift 2
fi

# recal calibration — MUST match training exactly. All FiLM calib buffers are
# persistent=False, so these envs are the ONLY source at deploy (no fallback in ckpt).
# NOTE: differs from doc §6.9 (old 0729 calib was F0=6/tau=4, FZ_OFF=2.1, no fmag).
RECAL_ENV=(FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1
           FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1
           FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7
           FILM_DATASET="$DIR/local_film_stats/lges_case_pick_0729")

# Both eval runs (--film-auto-baseline, task start) and the live probe
# (--baseline-hover, pre-descent hover) self-anchor against the live wrench,
# stashing the env values as the TRAIN offsets — so RECAL_ENV must stay at the
# training values and film_baseline_0729.env is NEVER applied here. The manual
# `baseline` target remains as a drift/pose diagnostic only (its env file is
# informational — the 08-04 run showed a hand-picked measurement pose can
# over-correct by ~0.7 N vs the actual task poses).

# match the July 0729 eval so results stay comparable (doc §6.7 command block).
# NOTE: run_policy appends <checkpoint_name>/L<layers> to --log-dir itself, so
# log-dir is a per-target root and the layer goes in via --layers.
COMMON=(--go --force-limit 15 --n-action-steps 5)
[[ -n "$LAYER" ]] && COMMON+=(--layers "${LAYER#L}")
FILM=(--film --film-auto-baseline)   # re-anchors offsets at every task start
FROMNAIVE=Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive
PM1R=Chanho-Lee/smolvla_film_0729_prefix_mask1_recal

case "$TARGET" in
  baseline)  # passive hover-force measurement -> film_baseline_0729.env (no motion)
    python measure_force_baseline.py "$@" ;;
  naive)  # same-day control (July logs live in rollouts/smolvla_naive_0729/)
    python run_policy.py --checkpoint Chanho-Lee/smolvla_naive_0729 \
      "${COMMON[@]}" --log-dir rollouts/naive_0729_ctrl0804 "$@" ;;
  fromnaive_best)  # main branch = val-best
    env "${RECAL_ENV[@]}" python run_policy.py "${FILM[@]}" --checkpoint "$FROMNAIVE" \
      "${COMMON[@]}" --log-dir rollouts/film_0729_fromnaive_best "$@" ;;
  fromnaive_last)  # 20k — wins ALL offline metrics (swap 97%, sim 1.7mm, err 0.81mm)
    env "${RECAL_ENV[@]}" python run_policy.py "${FILM[@]}" --checkpoint "$FROMNAIVE" \
      --revision last \
      "${COMMON[@]}" --log-dir rollouts/film_0729_fromnaive_last "$@" ;;
  pm1r_best)  # base-init A/B (does the naive warm-start matter on-robot?)
    env "${RECAL_ENV[@]}" python run_policy.py "${FILM[@]}" --checkpoint "$PM1R" \
      "${COMMON[@]}" --log-dir rollouts/film_0729_pm1_recal_best "$@" ;;
  pm1r_last)
    env "${RECAL_ENV[@]}" python run_policy.py "${FILM[@]}" --checkpoint "$PM1R" \
      --revision last \
      "${COMMON[@]}" --log-dir rollouts/film_0729_pm1_recal_last "$@" ;;
  live_probe)  # S3 on-robot counterfactual, film vs naive on the same frozen obs
    env "${RECAL_ENV[@]}" python probe_film_authority_live.py --go \
      --clearances 0.05 0.04 0.03 0.02 0.01 0.00 -0.01 -0.02 -0.03 -0.04 \
      --checkpoint "$FROMNAIVE" \
      --baseline-checkpoint Chanho-Lee/smolvla_naive_0729 \
      --fz-deltas-n -6 -3 3 6 "$@" ;;
  *) echo "unknown target '$TARGET'" >&2; exit 1 ;;
esac
