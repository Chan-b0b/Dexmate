#!/usr/bin/env bash
# 0909 causal-interpretation probes (2026-09-14). Two cells the existing battery cannot
# produce, both aimed at the claim "the FiLM path learned 'pushing back -> stop', it did
# not memorise the scene".
#
# E1  OFF-CONTACT INJECTION  (--pre-contact-offset)
#   Every authority cell in the round (pc_fc, pc_r12, the whole 12-55N sweep) evaluates
#   the 10 frames immediately BEFORE first contact. That window is exactly where visual
#   timing memorisation ALSO predicts "stop soon" — EXPERIMENTS §7 shows groot_naive has
#   already halted there (dz(own)=+0.30mm) on vision alone. So braking at offset 0 is
#   consistent with both hypotheses and discriminates nothing.
#   Injecting the SAME contact wrench 15 and 30 frames earlier — mid-descent, ee_z ~0.91-0.96
#   vs ~0.76-0.79, where contact never occurs in training — breaks the tie: only a force
#   law still brakes there. This is the open-loop counterpart of press-sim off30.
#
# E2  DIRECTION NEGATIVE CONTROL  (--swap fcflip / fcrand)
#   Every swap in the round raises force TOWARD contact, so nothing separates
#   "force up -> stop" from "conditioning perturbed -> freeze". fcflip negates the force
#   xyz and fcrand randomises it, both rescaled to the SAME |F|. Because
#   contact = clip((|F|-F0)/tau) depends on the norm only, the 'contact' channel is
#   bit-identical across fcscale/fcflip/fcrand — only 'fz' flips sign or scrambles.
#   A learned law must brake on fcscale and NOT on fcflip. Equal braking on both = the
#   response rides on |F| alone, or is a freeze reflex, and the causal reading dies.
#
# Both experiments are probes over the EXISTING mask=1 best checkpoints — no training.
# --fc-mag 16 throughout: inside the c-hat window (contact=0.85, not yet saturated at
# 16.83N) and the magnitude where §8's dose-response peaks for the SmolVLA FiLM arms.
#
# E2's fcscale reference cell at (mag 16, offset 0) already exists as
# probes/fsweep_<arm>_0909_16.txt and is NOT recomputed — verified bit-identical after the
# --pre-contact-offset patch.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
export HF_HOME="$HOME/.cache/huggingface"
RV="$DIR/datasets/lges_case_pick_0909_val"
REPO=Chanho-Lee/lges_case_pick_0909_val
V="$DIR/logs/validate_0909.out"
cal(){ grep -o "^CAL_$1=.*" "$V"|tail -1|cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZT="$(cal FZ_TAU)"; FZO="$(cal FZ_OFF)"
GPU=${GPU:-2}; JOBS=${JOBS:-4}; MAG=${MAG:-16}
# offset+10 must stay below the shortest first-contact index (0909 min = 48), else the
# window runs off the front of the episode and silently loses frames.
OFFSETS=${OFFSETS:-"0 15 30"}
MODES=${MODES:-"fcflip fcrand"}
mkdir -p "$DIR/probes" "$DIR/logs"
echo "[causal] cal F0=$F0 TAU=$TAU (contact saturates at $(echo "$F0+$TAU"|bc))  mag=$MAG gpu=$GPU jobs=$JOBS"

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

cell() { # arm flag inject out_name swap offset
  local a=$1 flag=$2 inj=$3 out="$DIR/probes/$4" swap=$5 off=$6
  [[ -s "$out" ]] && { echo "[causal] skip $4 (exists)"; return 0; }
  local FL=(); [[ "$flag" != "-" ]] && FL=("$flag")
  env FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_F0="$F0" FILM_TAU="$TAU" \
      FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZO" FILM_INJECT="$inj" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" "$DIR/probe_state_authority.py" ${FL[@]+"${FL[@]}"} \
      --checkpoint "$DIR/outputs/$a/checkpoints/best" \
      --dataset-root "$RV" --repo-id "$REPO" \
      --all-episodes --swap "$swap" --fc-mag "$MAG" \
      --pre-contact 10 --pre-contact-offset "$off" > "$out" 2>&1
  echo "[causal] $4 rc=$?"
}

n=0
for spec in "${ARMS[@]}"; do
  read -r a flag inj <<<"$spec"
  # E1: same wrench, window walked back from first contact
  for off in $OFFSETS; do
    cell "$a" "$flag" "$inj" "oc_${a}_${MAG}_off${off}.txt" fcscale "$off" &
    n=$((n+1)); (( n % JOBS == 0 )) && wait
  done
  # E2: same |F|, direction destroyed, at the original window
  for m in $MODES; do
    cell "$a" "$flag" "$inj" "nc_${a}_${MAG}_${m}.txt" "$m" 0 &
    n=$((n+1)); (( n % JOBS == 0 )) && wait
  done
done
wait
echo "[causal] DONE ($n cells)"
