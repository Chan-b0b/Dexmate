#!/usr/bin/env bash
# Re-run only the pc_fc cell for all 9 arms with the corrected --fc-fz-thresh (2026-09-11).
# At the probe's 3.0 default the "first force rise" trigger (fz > thresh for 2 frames) is
# true from frame 0 on 0909 (fz baseline 7.54N), so the swap wrench was pooled from
# early-descent frames and injected only |F|=8.6N — below F0=11.26, i.e. c-hat contact=0
# and fz channel -0.44. The cell was measuring "less load than baseline", and every FiLM
# arm correctly responded by descending MORE (negative dTotal). At the derived 10.0 the
# pool is the real first-contact frames, |F|=12.9N, c-hat=(0.293, +0.647, 0).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maverick/vla_venv/bin/python
export HF_HOME="$HOME/.cache/huggingface"
RV="$DIR/datasets/lges_case_pick_0909_val"; REPO=Chanho-Lee/lges_case_pick_0909_val
V="$DIR/logs/validate_0909.out"; cal(){ grep -o "^CAL_$1=.*" "$V"|tail -1|cut -d= -f2-; }
F0="$(cal F0)"; TAU="$(cal TAU)"; FZT="$(cal FZ_TAU)"; FZO="$(cal FZ_OFF)"; FCFZ="$(cal FC_FZ)"
GPU=${GPU:-2}; JOBS=${JOBS:-3}
echo "[pcfc] fc_fz_thresh=$FCFZ (probe default 3.0)  gpu=$GPU"
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
  ( FL=(); [[ "$flag" != "-" ]] && FL=("$flag")
    env FILM_COND=contact,fz,seal FILM_MASK_FORCE=1 FILM_F0="$F0" FILM_TAU="$TAU" \
        FILM_FZ_TAU="$FZT" FILM_FZ_OFF="$FZO" FILM_INJECT="$inj" CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" "$DIR/probe_state_authority.py" ${FL[@]+"${FL[@]}"} \
        --checkpoint "$DIR/outputs/$a/checkpoints/best" \
        --dataset-root "$RV" --repo-id "$REPO" --all-episodes \
        --swap firstcontact --pre-contact 10 --fc-fz-thresh "$FCFZ" \
        > "$DIR/probes/0909_${a}_pc_fc.txt" 2>&1
    echo "[pcfc] $a rc=$?" ) &
  n=$((n+1)); (( n % JOBS == 0 )) && wait
done
wait
echo "[pcfc] DONE ($n arms)"
