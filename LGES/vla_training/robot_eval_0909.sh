#!/usr/bin/env bash
# Robot eval — 0909 round. RUN THIS ON THE ROBOT HOST, not the B300 server.
#
# Offline evidence this is meant to test (EXPERIMENTS_CASE_PICK_0909):
#   groot_film_layers is the only arm that stops at BOTH contact timings (§5: off1 10/10
#   @13.4mm, off30 10/10 @16.7mm), the only one whose braking GROWS when the contact
#   wrench is injected far from contact (§8.5: +1.00 -> +3.19 mm/step), and the only one
#   that refuses the raw-state path when the mask is lifted (§10.1: dRaw = -0.07 while
#   pi0/smolvla route 83-100% through raw).
#   groot_naive is the control: best at normal timing (§5 off1 1.1mm) but 0/10 at off30,
#   and §8.5 shows its response is flat (+0.49 -> +0.57) = it stops on visual timing.
#
# THE PREDICTION TO FALSIFY: on early contact, groot_film_layers stops and groot_naive
# does not. If both stop, press-sim's stiffness model was doing the work. If neither
# does, the offline probes do not transfer.
#
# ---------------------------------------------------------------------------------
# STEP 0 FIRST, AND DO NOT SKIP IT. 0909's F0=11.26 / TAU=5.57 were DERIVED from the
# force distribution of 0909's F/T mounting (§3). The mount moved once already: 0816's
# F0=6 ended up BELOW 0909's hover force, so the contact channel read ~0.8 while hovering
# and discriminated nothing. If this robot's hover |F| is not near 10.5N, the 0909
# calibration is invalid here and every rollout below is measuring noise.
#   --film-auto-baseline re-anchors the OFFSET against the live wrench at task start.
#   It cannot fix a changed SCALE: F0/TAU come from the contact force distribution.
# ---------------------------------------------------------------------------------
#
# Usage: ./robot_eval_0909.sh <target> [extra run_policy args...]
#   baseline           STEP 0. passive hover-force measurement, NO MOTION. Start here.
#   groot_naive        control arm
#   groot_film_layers  the arm the paper's claim rests on
#   groot_film_state   second GR00T FiLM variant (§5 off30 2/10)
#   live_probe         on-robot counterfactual (film vs naive on the same frozen obs)
#
# Prereqs:
#   1. GR00T deploy support: run_policy.py now accepts --film on groot checkpoints
#      (auto-detects inject from the checkpoint, validates len(FILM_COND) against
#      contact_film's Linear, imports train_groot for the sdpa fallback).
#   2. Checkpoints reachable. Either local dirs copied from the server, or HF repo ids
#      via CKPT_ROOT. GR00T needs NO observation-key remapping: it trained on the raw
#      dataset so its input_features are observation.images.head / head_depth /
#      observation.state, exactly what run_policy.predict() builds.
#      (The 0909 SmolVLA arms went through train_film.sh's --rename_map to
#      camera1/camera2 and therefore do NOT match run_policy's keys — they need a
#      rename layer before they can be deployed. Not done here.)
#   3. local_film_stats/lges_case_pick_0909/meta/stats.json present (GR00T min-max
#      normalizes state, so the deploy path needs this dataset's stats; the datasets/
#      symlink is broken on the robot).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

TARGET=${1:?usage: $0 <target> [run_policy args...]}
shift

# 0909 calibration — MUST match training exactly. All FiLM calib buffers are
# persistent=False, so these envs are the ONLY source at deploy (no fallback in the
# checkpoint). Values are the CAL_* the validate gate derived; do not hand-edit.
CAL_ENV=(FILM_COND=contact,fz,seal FILM_MASK_FORCE=1
         FILM_F0=11.26 FILM_TAU=5.57 FILM_FZ_OFF=8.18 FILM_FZ_TAU=3.71
         FILM_DATASET=lges_case_pick_0909)
# FILM_INJECT is deliberately NOT set: run_policy auto-detects it from the checkpoint's
# own tensors (indexed contact_film.<i>.* => layers) and errors out if an env contradicts
# the weights. Structural, so it cannot silently mismatch.

CKPT_ROOT=${CKPT_ROOT:-$DIR/outputs}
ck() { echo "$CKPT_ROOT/$1_0909/checkpoints/best"; }

# --force-limit 12, not the 0729 round's 15: §6 derives the force spec from the
# demonstrations' settled press (14.09N p95) x1.5 = 21N as the damage threshold, and the
# six arms that stop on-policy all peak at 8-17N. 12 aborts before the spec, leaving
# headroom on the first runs. Raise deliberately, never as a reflex.
COMMON=(--go --force-limit 12 --n-action-steps 5)

case "$TARGET" in
  baseline)   # STEP 0 — passive, no motion. Compare hover |F| against 0909's 10.51N.
    python measure_force_baseline.py "$@" ;;
  groot_naive)
    python run_policy.py --checkpoint "$(ck groot_naive)" \
      "${COMMON[@]}" --log-dir rollouts/0909_groot_naive "$@" ;;
  groot_film_layers)
    env "${CAL_ENV[@]}" python run_policy.py --film --film-auto-baseline \
      --checkpoint "$(ck groot_film_layers)" \
      "${COMMON[@]}" --log-dir rollouts/0909_groot_film_layers "$@" ;;
  groot_film_state)
    env "${CAL_ENV[@]}" python run_policy.py --film --film-auto-baseline \
      --checkpoint "$(ck groot_film_state)" \
      "${COMMON[@]}" --log-dir rollouts/0909_groot_film_state "$@" ;;
  live_probe)  # on-robot counterfactual: film vs naive on the SAME frozen observation.
    # The closest on-robot analogue of §8.5 — --clearances sweeps the hold height, so
    # the large clearances are the off-contact window that separates a force law from
    # visual timing.
    # !! NOT USABLE YET: probe_film_authority_live.py contains zero groot references --
    # it was written for the 0729 smolvla arms and has no --film-groot path. It needs the
    # same treatment run_policy.py just got (film_contact_groot patch + owner resolution
    # on the policy rather than policy.model) before this target will run.
    env "${CAL_ENV[@]}" python probe_film_authority_live.py --go \
      --clearances 0.05 0.04 0.03 0.02 0.01 0.00 \
      --checkpoint "$(ck groot_film_layers)" \
      --baseline-checkpoint "$(ck groot_naive)" \
      --fz-deltas-n -6 -3 3 6 "$@" ;;
  *) echo "unknown target '$TARGET'" >&2; exit 1 ;;
esac
