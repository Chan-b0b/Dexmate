#!/usr/bin/env bash
# 0909 stop-threshold sweep (2026-09-11): at the REAL touch pose, raise |F| and find the
# force at which each policy stops commanding descent.
#
# Why not press_sim: that probe is closed-loop, so penetration accumulates and three
# artefacts dominate — |F| pegs at --f-cap 25 (every 0909 arm ended at 33.3-33.8N, zero
# resolution), --stiffness 1.0 is not matched to 0909's 2x larger action scale, and
# --max-steps 40 turns any residual creep into 40mm of depth. Holding the pose fixed and
# sweeping |F| removes all three and yields the number directly: "this policy needs X N".
#
# Cell: --swap fcscale --fc-mag X --pre-contact 10 --all-episodes
#   = the 10 real frames before each episode's first contact (pose + image untouched),
#     wrench replaced by the first-contact DIRECTION rescaled to |F| = X.
#   --pre-contact is essential: without it the default "committed descent" frame filter
#   matches only n=3 frames on 0909 (baseline |F| is 10.5N, not ~5N), which is why the
#   battery's ramp8/10/12 cells were uninformative this round.
#
# Read the COMMITTED-descent row, not ALL frames: the ALL pool mixes in 56 hover frames
# whose dz is already ~0 and which fake an apparent stop.
#
# NOTE on interpretation: the FiLM arms cannot see past F0+TAU = 16.83N (c-hat clips to
# 1.0), so above that point naive keeps receiving new information and FiLM does not. The
# sweep therefore asks "does FiLM arrest inside its own 11.26-16.83N window?", not "who
# wins at 40N".
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
export HF_HOME="$HOME/.cache/huggingface"
RV="$DIR/datasets/lges_case_pick_0909_val"
REPO=Chanho-Lee/lges_case_pick_0909_val
V="$DIR/logs/validate_0909.out"
cal(){ grep -o "^CAL_$1=.*" "$V"|tail -1|cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZT="$(cal FZ_TAU)"; FZO="$(cal FZ_OFF)"
GPU=${GPU:-3}; JOBS=${JOBS:-4}
MAGS=${MAGS:-"12 14 16 18 20 24 28 34 40"}
mkdir -p "$DIR/probes" "$DIR/logs"
echo "[fsweep] cal F0=$F0 TAU=$TAU (saturates at $(echo "$F0+$TAU"|bc))  gpu=$GPU jobs=$JOBS mags=$MAGS"

#      arm                        flag          inject
ARMS=( "pi0_naive_0909            --naive       state"
       "pi0_film_state_0909       --film-pi0    state"
       "pi0_film_layers_0909      --film-pi0    layers"
       "groot_naive_0909          --naive       state"
       "groot_film_state_0909     --film-groot  state"
       "groot_film_layers_0909    --film-groot  layers"
       "smolvla_naive_0909        --naive       state"
       "smolvla_film_state_0909   -             prefix"
       "smolvla_film_layers_0909  -             layers" )

run_cell() { # arm flag inject mag
  local a=$1 flag=$2 inj=$3 m=$4
  local out="$DIR/probes/fsweep_${a}_${m}.txt"
  [[ -s "$out" ]] && { echo "[fsweep] skip $a @${m}N (exists)"; return 0; }
  local FL=(); [[ "$flag" != "-" ]] && FL=("$flag")
  env FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_F0="$F0" FILM_TAU="$TAU" \
      FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZO" FILM_INJECT="$inj" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" "$DIR/probe_state_authority.py" ${FL[@]+"${FL[@]}"} \
      --checkpoint "$DIR/outputs/$a/checkpoints/best" \
      --dataset-root "$RV" --repo-id "$REPO" \
      --all-episodes --swap fcscale --fc-mag "$m" --pre-contact 10 > "$out" 2>&1
  echo "[fsweep] $a @${m}N rc=$?"
}

n=0
for spec in "${ARMS[@]}"; do
  read -r a flag inj <<<"$spec"
  for m in $MAGS; do
    run_cell "$a" "$flag" "$inj" "$m" &
    n=$((n+1))
    (( n % JOBS == 0 )) && wait
  done
done
wait
echo "[fsweep] DONE ($n cells)"
