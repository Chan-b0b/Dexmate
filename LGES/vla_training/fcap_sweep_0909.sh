#!/usr/bin/env bash
# press-sim with the force cap raised (2026-09-11). At the derived f_cap=27.09 the failing
# arms all pegged at |F|=35.5N (the cap applies to the fz delta; |F| is the norm, so fx/fy
# push it higher), leaving the force metric with no resolution — and groot_film_layers'
# off30 peak of 26.7N sat right at the cap, so even the one working arm may have been
# clipped. Raising the cap trades physical realism for resolution: the injected fz goes far
# past the demonstrations' own maximum contact force (26.6N), so any policy response above
# that is extrapolation outside the training distribution. Recorded deliberately.
# Everything else is the battery's press-sim cell verbatim.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
export HF_HOME="$HOME/.cache/huggingface"
RV="$DIR/datasets/lges_case_pick_0909_val"; REPO=Chanho-Lee/lges_case_pick_0909_val
V="$DIR/logs/validate_0909.out"; cal(){ grep -o "^CAL_$1=.*" "$V"|tail -1|cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZT="$(cal FZ_TAU)"; FZO="$(cal FZ_OFF)"; FZD="$(cal FZ_DELTA0)"
FBASE="$(cal F_BASE)"
FCAP=${FCAP:-100}; GPU=${GPU:-3}; JOBS=${JOBS:-4}
echo "[fcap] f_cap=$FCAP (derived was $(cal F_CAP); demo max contact |F| = 26.64N)"
ARMS=( "pi0_naive_0909            --naive       state"
       "pi0_film_state_0909       --film-pi0    state"
       "pi0_film_layers_0909      --film-pi0    layers"
       "groot_naive_0909          --naive       state"
       "groot_film_state_0909     --film-groot  state"
       "groot_film_layers_0909    --film-groot  layers"
       "smolvla_naive_0909        --naive       state"
       "smolvla_film_state_0909   -             prefix"
       "smolvla_film_layers_0909  -             layers" )
n=0
for s in "${ARMS[@]}"; do
  read -r a flag inj <<<"$s"
  for off in 1 30; do
    ( FL=(); [[ "$flag" != "-" ]] && FL=("$flag")
      env FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_F0="$F0" FILM_TAU="$TAU" \
          FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZO" FILM_INJECT="$inj" CUDA_VISIBLE_DEVICES="$GPU" \
        "$PY" "$DIR/probe_press_sim.py" ${FL[@]+"${FL[@]}"} \
          --checkpoint "$DIR/outputs/$a/checkpoints/best" \
          --dataset-root "$RV" --repo-id "$REPO" \
          --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
          --fz-delta0 "$FZD" --f-base "$FBASE" --f-cap "$FCAP" \
          > "$DIR/probes/0909fc${FCAP}_sim_${a}_off${off}.txt" 2>&1
      echo "[fcap] $a off$off rc=$?" ) &
    n=$((n+1)); (( n % JOBS == 0 )) && wait
  done
done
wait
echo "[fcap] DONE ($n cells)"
