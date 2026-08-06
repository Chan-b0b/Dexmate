#!/usr/bin/env bash
# Generic offline probe battery — HF checkpoint vs local 0729 val set (LOCAL machine).
# Usage: ./probe_0729_offline.sh <hf_repo> <tag> [mask=1] [kind=film|naive] [revision=main]
#   e.g. ./probe_0729_offline.sh Chanho-Lee/smolvla_film_0729_prefix_mask0_recal_fromnaive mask0fn 0
#        ./probe_0729_offline.sh Chanho-Lee/pi05_naive_0729 pi05naive 1 naive
#        ./probe_0729_offline.sh Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive fromnaive 1 film last
# Battery (decisive-first): state-swap pc_fc -> force-scale ramp8/12 -> pc_r12
# -> std authority (film only) -> press-sim fzdelta off1/30 -> eval_offline.
# Outputs: probes/0729_*_<tag>_<revision>_*.txt
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python
REPO=${1:?hf repo}; TAG=${2:?tag}; MASK=${3:-1}; KIND=${4:-film}; REV=${5:-main}
SUF="${TAG}_${REV}"
RV="$DIR/datasets_local/lges_case_pick_0729_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val

# recal calibration env (0729 recal generation; harmless/unused for naive ckpts)
FENV=(FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix "FILM_MASK_FORCE=$MASK"
      FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1
      FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7)
NAIVE_FLAG=(); [[ "$KIND" == naive ]] && NAIVE_FLAG=(--naive)

[[ -d "$RV/meta" ]] || { echo "[$TAG] val dataset missing" >&2; exit 1; }
mkdir -p "$DIR/probes" "$DIR/local_ckpts"
snap=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO', revision='$REV'))") \
  || { echo "[$TAG] HF fetch failed ($REPO @ $REV)" >&2; exit 1; }
mkdir -p "$DIR/local_ckpts/$SUF"
ln -sfn "$snap" "$DIR/local_ckpts/$SUF/pretrained_model"
CK="$DIR/local_ckpts/$SUF"
echo "[$TAG] ckpt: $snap (rev=$REV)"

state() { # <outfile> <extra args...>
  local out=$1; shift
  env "${FENV[@]}" "$PY" "$DIR/probe_state_authority.py" "${NAIVE_FLAG[@]}" \
    --checkpoint "$CK" --dataset-root "$RV" --repo-id "$REPO_VAL" --all-episodes "$@" \
    > "$DIR/probes/$out" 2>&1
  echo "[$TAG] $out rc=$? -> $(grep -a -m1 'ALL frames' "$DIR/probes/$out" | sed 's/^ *//')"
}

state "0729_state_${SUF}_pc_fc.txt"  --swap firstcontact --pre-contact 10
state "0729_state_${SUF}_ramp8.txt"  --swap fcscale --fc-mag 8
state "0729_state_${SUF}_ramp12.txt" --swap fcscale --fc-mag 12
state "0729_state_${SUF}_pc_r12.txt" --swap fcscale --fc-mag 12 --pre-contact 10

if [[ "$KIND" != naive ]]; then
  env "${FENV[@]}" "$PY" "$DIR/probe_film_authority.py" --checkpoint "$CK" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
    > "$DIR/probes/0729_${SUF}_std.txt" 2>&1; echo "[$TAG] std rc=$?"
fi
for off in 1 30; do
  env "${FENV[@]}" "$PY" "$DIR/probe_press_sim.py" "${NAIVE_FLAG[@]}" --checkpoint "$CK" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" \
    --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
    > "$DIR/probes/0729_sim_${SUF}_off${off}_fzdelta.txt" 2>&1
  echo "[$TAG] sim off$off rc=$?"
done
EVF=(--film); [[ "$KIND" == naive ]] && EVF=()
env "${FENV[@]}" "$PY" "$DIR/eval_offline.py" "${EVF[@]}" --checkpoint "$CK" \
  --val-root "$RV" --repo-id "$REPO_VAL" \
  > "$DIR/probes/0729_eval_${SUF}.txt" 2>&1; echo "[$TAG] eval rc=$?"

echo; echo "===== $TAG verdict material ====="
grep -a -H "ALL frames" "$DIR"/probes/0729_state_${SUF}_*.txt
grep -a -H -A3 "── summary" "$DIR"/probes/0729_sim_${SUF}_*.txt | grep -v "^--$"
grep -a -H "OVERALL" "$DIR"/probes/0729_eval_${SUF}.txt
echo "[$TAG] DONE"
