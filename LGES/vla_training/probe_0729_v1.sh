#!/usr/bin/env bash
# V1 (decorrelated control) probe battery — LOCAL version (robot machine):
# pulls weights from HF, probes against datasets_local val. Decisive probes first.
#
# THE decisive test for the "form is determined by design" claim (DISCUSSION_LOG
# 08-04, EVIDENCE §3.5): V1 = same init (naive best), same data, same capacity,
# c-hat SHUFFLED at train time -> grounding removed. Prediction: the force-scale
# sweep (ramp8/ramp12) comes out FLAT or naive-like-declining. If V1 shows the
# monotone rise (fromnaive was +1.20 -> +4.40mm), the claim is falsified.
#
# Probes hardcode <ckpt>/pretrained_model (server layout), so HF snapshots get a
# symlink wrapper under local_ckpts/ — no probe-code changes.
# Runtime on the Jetson: sanity+ramps ~40min, full battery ~2h.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python
V1_REPO=${1:-Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive_v1}   # main only
FN_REPO=Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive           # pipeline check
RV="$DIR/datasets_local/lges_case_pick_0729_val"
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val

# must match V1 TRAINING env (recal calibration; the v1 shuffle is train-time only)
RECAL_ENV=(FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1
           FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1
           FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7)

[[ -d "$RV/meta" ]] || { echo "[v1] val dataset missing at $RV" >&2; exit 1; }
mkdir -p "$DIR/probes" "$DIR/local_ckpts"

fetch() {  # <repo> <name> -> local_ckpts/<name>/ with pretrained_model -> HF snapshot
  local snap
  snap=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))") || return 1
  mkdir -p "$DIR/local_ckpts/$2"
  ln -sfn "$snap" "$DIR/local_ckpts/$2/pretrained_model"
  echo "$DIR/local_ckpts/$2"
}

echo "[v1] fetching weights from HF..."
V1CK=$(fetch "$V1_REPO" v1_main) || { echo "[v1] fetch failed: $V1_REPO" >&2; exit 1; }
FNCK=$(fetch "$FN_REPO" fromnaive_main) || { echo "[v1] fetch failed: $FN_REPO" >&2; exit 1; }

state() {  # <ckpt> <outfile> <extra args...>
  local ck=$1 out=$2; shift 2
  env "${RECAL_ENV[@]}" "$PY" "$DIR/probe_state_authority.py" \
    --checkpoint "$ck" --dataset-root "$RV" --repo-id "$REPO_VAL" --all-episodes "$@" \
    > "$DIR/probes/$out" 2>&1
  echo "[v1] $out rc=$? -> $(grep -a -m1 'ALL frames' "$DIR/probes/$out" | sed 's/^ *//')"
}

# 0) pipeline sanity: local rerun of fromnaive pc_fc must reproduce server +1.57mm (94%)
state "$FNCK" 0729_state_fromnaive_LOCALCHECK_pc_fc.txt --swap firstcontact --pre-contact 10

# 1) DECISIVE: V1 force-scale sweep (all-frames n=245 convention, matches naive/fromnaive ramps)
state "$V1CK" 0729_state_v1_main_ramp8.txt  --swap fcscale --fc-mag 8
state "$V1CK" 0729_state_v1_main_ramp12.txt --swap fcscale --fc-mag 12
state "$V1CK" 0729_state_v1_main_pc_fc.txt  --swap firstcontact --pre-contact 10
state "$V1CK" 0729_state_v1_main_pc_r12.txt --swap fcscale --fc-mag 12 --pre-contact 10

# 2) supporting: action-accuracy equivalence + std authority + closed-loop press sim
env "${RECAL_ENV[@]}" "$PY" "$DIR/eval_offline.py" --film --checkpoint "$V1CK" \
  --val-root "$RV" --repo-id "$REPO_VAL" \
  > "$DIR/probes/0729_eval_v1_main.txt" 2>&1; echo "[v1] eval rc=$?"
env "${RECAL_ENV[@]}" "$PY" "$DIR/probe_film_authority.py" --checkpoint "$V1CK" \
  --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
  > "$DIR/probes/0729_v1_main_std.txt" 2>&1; echo "[v1] std rc=$?"
for off in 1 30; do
  env "${RECAL_ENV[@]}" "$PY" "$DIR/probe_press_sim.py" --checkpoint "$V1CK" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" \
    --stiffness 1.0 --seal-depth 0 --start-offset "$off" --force-model fzdelta \
    > "$DIR/probes/0729_sim_v1_main_off${off}_fzdelta.txt" 2>&1
  echo "[v1] sim off$off rc=$?"
done

echo; echo "===== V1 verdict material ====="
grep -a -H "ALL frames" "$DIR"/probes/0729_state_v1_main_*.txt \
  "$DIR"/probes/0729_state_fromnaive_LOCALCHECK_pc_fc.txt
grep -a -H -A3 "── summary" "$DIR"/probes/0729_sim_v1_main_*.txt | grep -v "^--$"
grep -a -H "OVERALL" "$DIR"/probes/0729_eval_v1_main.txt
echo "[v1] DONE"
