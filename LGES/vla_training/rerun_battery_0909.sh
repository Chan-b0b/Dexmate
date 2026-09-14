#!/usr/bin/env bash
# Re-run the 0909 probe battery for all 9 arms after the 2026-09-11 fixes (2026-09-11).
# Training is untouched — only the evaluation was wrong. Four 0816-era constants had been
# hardcoded and were all invalidated by this collection's F/T remount (+5.4N baseline):
#   FZ_DELTA0 1.7 -> 1.17     measured fz jump at touch
#   F_BASE    6.8 -> 18.43    measured |F| at touch
#   F_CAP     25  -> 27.09    1.3*max(episode contact peak) - pre-touch fz median
#   FZ_TAU    5   -> 3.71     was hardcoded in the battery's FENV even after F0/TAU were not
# plus PRE_CONTACT=10 for the ramp cells: the |F| < --descend-n 6N frame filter matched
# n=3 frames on 0909 (and none of them pre-contact), so ramp8/10/12 measured wrong frames.
# And probe_{state_authority,press_sim}.py's touch detector was replaced — the old
# |ΔF|>=2N rule misfired on 2 of 10 val episodes (reporting touch at z=1.13-1.18m instead
# of 0.81m), which is where press-sim's 430-480mm "catastrophic" outliers came from.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RV="$DIR/datasets/lges_case_pick_0909_val"
V="$DIR/logs/validate_0909.out"
cal(){ grep -o "^CAL_$1=.*" "$V"|tail -1|cut -d= -f2-; }
GPU=${GPU:-3}; JOBS=${JOBS:-3}
export VAL_ROOT="$RV" REPO_VAL=Chanho-Lee/lges_case_pick_0909_val PFX=0909
export FZ_OFF="$(cal FZ_OFF)" FZ_TAU="$(cal FZ_TAU)" F0="$(cal F0)" TAU="$(cal TAU)" C1="$(cal C1)"
export FZ_DELTA0="$(cal FZ_DELTA0)" F_BASE="$(cal F_BASE)" F_CAP="$(cal F_CAP)" \
       PRE_CONTACT="$(cal PRE_CONTACT)" FC_FZ="$(cal FC_FZ)"
echo "[rerun] F0=$F0 TAU=$TAU FZ_OFF=$FZ_OFF FZ_TAU=$FZ_TAU C1=$C1"
echo "[rerun] fz_delta0=$FZ_DELTA0 f_base=$F_BASE f_cap=$F_CAP pre_contact=$PRE_CONTACT fc_fz=$FC_FZ"

ARMS=( "pi0_naive_0909            naive       -"
       "pi0_film_state_0909       film-pi0    state"
       "pi0_film_layers_0909      film-pi0    layers"
       "groot_naive_0909          naive       -"
       "groot_film_state_0909     film-groot  state"
       "groot_film_layers_0909    film-groot  layers"
       "smolvla_naive_0909        naive       -"
       "smolvla_film_state_0909   film        prefix"
       "smolvla_film_layers_0909  film        layers" )
n=0
for s in "${ARMS[@]}"; do
  read -r name kind inject <<<"$s"
  ( "$DIR/run_battery_0729cal.sh" "$name" "$kind" "$inject" \
      "$DIR/outputs/$name/checkpoints/best" "$GPU" > "$DIR/logs/rebat_${name}.out" 2>&1
    echo "[rerun] $name rc=$?" ) &
  n=$((n+1)); (( n % JOBS == 0 )) && wait
done
wait
echo "[rerun] DONE ($n arms)"
