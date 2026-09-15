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
#   baseline                    STEP 0. passive hover-force measurement, NO MOTION. Start here.
#   smolvla_naive               control arm (needs no FiLM env at all)
#   smolvla_film_state          prefix injection — one film on the state token
#   smolvla_film_layers         per-layer injection — the arm the paper's claim rests on
#   smolvla_film_layers_nomask  same arm, mask lifted (FILM_MASK_FORCE=0)
#   groot_naive                 control arm
#   groot_film_layers           per-DiT-block injection — the arm the paper's claim rests on
#   groot_film_state            state-token injection at the action head's state_encoder
#   live_probe                  on-robot counterfactual (film vs naive on the same frozen obs)
#
# ALWAYS pass --layers 1 or --layers 5. The default is cfg.SRC_LAYERS_REMAINING=3, and the
# 0909 collection only ever recorded 1-layer (60 takes) and 5-layer (50 takes) stacks — a
# 2/3/4-high stack is a scene no 0909 arm has seen. Set the physical stack to 1 or 5 and
# say which; --layers only feeds the BEV warp plane and the log, never the policy.
#
# Prereqs:
#   1. FiLM deploy support covers SMOLVLA and GR00T (2026-09-14). run_policy auto-detects
#      the injection layout from the checkpoint's own tensors, so FILM_INJECT is never
#      needed and cannot silently mismatch:
#        smolvla — flat contact_film.* = prefix/suffix (told apart by width, 960 vs 720),
#                  indexed contact_film.<i>.* = layers.
#        groot   — keys carry NO 'model.' prefix (the film lives on the policy, not on
#                  .model) and width does not separate the two injections (DiT ff branch
#                  and state token are both 1536), so the indexed keys are the only
#                  discriminator: contact_film.<i>.* = layers, flat = state.
#      FILM_COND IS REQUIRED for groot and its length is validated against the film's
#      first Linear (cond_dim). Unlike smolvla, groot cannot fall back to auto-detecting
#      the cond NAMES — its calib buffers are persistent=False so they are not in the
#      file. (Auto-detection is unreliable on smolvla too: omitting FILM_COND there
#      detects cond=('seal',) and dies on a shape mismatch. Always pass it.)
#      groot_naive is unaffected (no FiLM).
#   2. Checkpoints reachable under CKPT_ROOT (default: /home/dexmate/ckpt_0909 when it
#      exists, else outputs/). Local dirs or HF repo ids both work. NEITHER backbone needs
#      observation-key remapping: GR00T trained on the raw dataset keys, and the SmolVLA
#      checkpoints carry their own rename_observations_processor (head -> camera1,
#      head_depth -> camera2) inside policy_preprocessor.json, so the keys
#      run_policy.predict() builds land correctly. Their input_features also list a
#      camera3 that has no rename — it was declared-but-absent in training too, so nothing
#      needs to supply it.
#   3. local_film_stats/lges_case_pick_0909/meta/stats.json present — both the GR00T path
#      (min-max state normalization) and every --film run (FILM_DATASET resolves here for
#      the wrench/seal stats) read it; the datasets/ symlink is broken on the robot.
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
#
# FILM_MASK_FORCE is the one setting that has NO footprint in the weights, so nothing can
# check it — the nomask target below overrides it to 0, everything else trains at 1.

# Bare `python` on the robot is /usr/bin/python and has no lerobot; the deploy venv does.
PY="${PY:-/home/dexmate/vla_venv/bin/python}"
# The SmolVLM2 processor config is cached locally but load_policy still HEADs huggingface.co
# for it; the robot's egress fails SSL and burns >2 min in retries before falling back.
export HF_HUB_OFFLINE=1

CKPT_ROOT=${CKPT_ROOT:-$([[ -d /home/dexmate/ckpt_0909 ]] && echo /home/dexmate/ckpt_0909 || echo "$DIR/outputs")}
ck() { # resolve an arm name to its checkpoint dir, failing LOUDLY if it is not there:
  # run_policy treats a non-existent path as a Hub repo id, so a missing copy surfaces as
  # "HFValidationError: Repo id must be in the form 'repo_name'..." instead of "not found".
  # RETURNS (never exits): inside $( ) an exit would only kill the subshell and hand the
  # caller an empty path. Callers must assign it as a statement -- `CK=$(ck x)` -- so
  # set -e sees the failure.
  local p="$CKPT_ROOT/$1_0909/checkpoints/best"
  [[ -d "$p" ]] || { echo "[0909] checkpoint missing: $p" >&2
    echo "[0909] under $CKPT_ROOT: $(ls "$CKPT_ROOT" 2>/dev/null | tr '\n' ' ')" >&2
    echo "[0909] copy it from the B300 server, or set CKPT_ROOT." >&2; return 1; }
  echo "$p"; }

# --force-limit 12, not the 0729 round's 15: §6 derives the force spec from the
# demonstrations' settled press (14.09N p95) x1.5 = 21N as the damage threshold, and the
# six arms that stop on-policy all peak at 8-17N. 12 aborts before the spec, leaving
# headroom on the first runs. Raise deliberately, never as a reflex.
COMMON=(--go --force-limit 12 --n-action-steps 5)

# Resolve the checkpoint as a STATEMENT before the case (every arm target is named after
# its checkpoint dir), so set -e aborts on a missing copy with ck()'s message.
case "$TARGET" in       # arm targets only, so a typo still reaches "unknown target" below
  smolvla_naive|smolvla_film_state|smolvla_film_layers|smolvla_film_layers_nomask|\
  groot_naive|groot_film_layers|groot_film_state) CK=$(ck "$TARGET") ;;
esac

case "$TARGET" in
  baseline)   # STEP 0 — passive, no motion. Compare hover |F| against 0909's 10.51N.
    "$PY" measure_force_baseline.py "$@" ;;
  smolvla_naive)
    "$PY" run_policy.py --checkpoint "$CK" \
      "${COMMON[@]}" --log-dir rollouts/0909_smolvla_naive "$@" ;;
  smolvla_film_state)
    env "${CAL_ENV[@]}" "$PY" run_policy.py --film --film-auto-baseline \
      --checkpoint "$CK" \
      "${COMMON[@]}" --log-dir rollouts/0909_smolvla_film_state "$@" ;;
  smolvla_film_layers)
    env "${CAL_ENV[@]}" "$PY" run_policy.py --film --film-auto-baseline \
      --checkpoint "$CK" \
      "${COMMON[@]}" --log-dir rollouts/0909_smolvla_film_layers "$@" ;;
  smolvla_film_layers_nomask)  # trained with the mask lifted — the raw wrench stays in the
    # action path, so c-hat is added ON TOP instead of being the only route (run_nomask_0909.sh).
    env "${CAL_ENV[@]}" FILM_MASK_FORCE=0 "$PY" run_policy.py --film --film-auto-baseline \
      --checkpoint "$CK" \
      "${COMMON[@]}" --log-dir rollouts/0909_smolvla_film_layers_nomask "$@" ;;
  groot_naive)
    "$PY" run_policy.py --checkpoint "$CK" \
      "${COMMON[@]}" --log-dir rollouts/0909_groot_naive "$@" ;;
  groot_film_layers|groot_film_state)
    # Wired up 2026-09-14: load_policy routes groot to film_contact_groot (INJECTS =
    # state|layers), which un-normalizes c-hat with the dataset's state min/max because
    # groot min-max normalizes observation.state in groot_pack_inputs_v3 — not the
    # mean/std + seal stats the smolvla port uses.
    # FILM_BASELINE_ANCHOR_* below matter here: --film-auto-baseline anchors against the
    # COLLECTION's pre-touch force, and 0909 remounted the F/T (10.51/7.54, not 0729's
    # 4.59/1.96). Without them the drift is mis-read by ~6 N and contact_F0 lands above
    # every force the run can reach — the channel sits at 0 for the whole rollout.
    env "${CAL_ENV[@]}" "$PY" run_policy.py --film --film-auto-baseline \
      --checkpoint "$CK" \
      "${COMMON[@]}" --log-dir "rollouts/0909_$TARGET" "$@" ;;
  live_probe)  # on-robot counterfactual: film vs naive on the SAME frozen observation.
    # The closest on-robot analogue of §8.5 — --clearances sweeps the hold height, so
    # the large clearances are the off-contact window that separates a force law from
    # visual timing.
    # !! NOT USABLE YET: probe_film_authority_live.py contains zero groot references --
    # it was written for the 0729 smolvla arms and has no --film-groot path. It needs the
    # same treatment run_policy.py just got (film_contact_groot patch + owner resolution
    # on the policy rather than policy.model) before this target will run.
    CKF=$(ck groot_film_layers); CKN=$(ck groot_naive)
    env "${CAL_ENV[@]}" "$PY" probe_film_authority_live.py --go \
      --clearances 0.05 0.04 0.03 0.02 0.01 0.00 \
      --checkpoint "$CKF" \
      --baseline-checkpoint "$CKN" \
      --fz-deltas-n -6 -3 3 6 "$@" ;;
  *) echo "unknown target '$TARGET'" >&2; exit 1 ;;
esac
