#!/usr/bin/env bash
# 0729 round (model_list.txt): naive / film prefix_mask1 / film prefix_mask0 /
# film suffix_mask1 — cond=(contact,fz,seal), 50k, GPU 5.
# First round with a VAL split (Chanho-Lee/lges_case_pick_0729_val):
#   - retention: last + val-best (select_best_ckpt.py --prune)
#   - probes run on VAL episodes (generalization evidence)
# Storage: everything through $DIR symlinks -> /data (ABSOLUTE paths only; the
# validation step asserts output_dir stays absolute).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=5
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0729
REPO_VAL=Chanho-Lee/lges_case_pick_0729_val
RT="$DIR/datasets/lges_case_pick_0729"
RV="$DIR/datasets/lges_case_pick_0729_val"
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'

# ---- 1. download train + val ---------------------------------------------------
rm -rf "$RT" "$RV"
"$PY" -c "from huggingface_hub import snapshot_download as s; s('$REPO', repo_type='dataset', local_dir='$RT'); s('$REPO_VAL', repo_type='dataset', local_dir='$RV')"

# ---- 2. validate schema + force profile; measure fz median ---------------------
"$PY" - <<EOF | tee "$DIR/logs/validate_0729.out"
import glob, json
import numpy as np, pandas as pd
for tag, root in [("train", "$RT"), ("val", "$RV")]:
    info = json.load(open(f"{root}/meta/info.json"))
    f = info["features"]
    st = list(f["observation.state"]["shape"]); act = list(f["action"]["shape"])
    print(f"[0729] {tag}: eps={info['total_episodes']} frames={info['total_frames']} state={st} action={act}")
    assert st == [15], f"{tag} state {st} != [15]"
hover_F, press_F, all_fz, contact_z = [], [], [], []
for p in sorted(glob.glob("$RT/data/*/*.parquet")):
    df = pd.read_parquet(p, columns=["observation.state", "episode_index"])
    stt = np.stack(df["observation.state"].to_numpy())
    all_fz.append(stt[:, 11])
    for ep in df["episode_index"].unique():
        s = stt[(df["episode_index"] == ep).to_numpy()]
        fmag = np.linalg.norm(s[:, 9:12], axis=1)
        hover_F.append(np.median(fmag[:15]))
        seal = np.flatnonzero(s[:, 8] > 0.5)
        if len(seal): press_F.append(np.median(fmag[seal]))
        j = np.flatnonzero(np.abs(np.diff(fmag, prepend=fmag[0])) >= 2)
        if len(j): contact_z.append(s[j[0], 2])
hF, pF = np.median(hover_F), np.median(press_F)
fzm = float(np.median(np.concatenate(all_fz)))
cz = np.array(contact_z)
print(f"[0729] |F| hover={hF:.1f}N sealed={pF:.1f}N ({'RISE' if pF>hF else 'DROP'})  FZ_MEDIAN={fzm:.1f}")
print(f"[0729] contact z p10/50/90={np.percentile(cz,[10,50,90]).round(3)}")
assert pF > hF and hF < 6.0 < pF, f"force profile off (hover {hF:.1f}, press {pF:.1f}) — recalibrate F0"
print("[0729] validation OK")
EOF
[[ ${PIPESTATUS[0]} == 0 ]] || { echo "[0729] VALIDATION FAILED — not training" >&2; exit 1; }
FILM_FZ_OFF="$(grep -o 'FZ_MEDIAN=[0-9.-]*' "$DIR/logs/validate_0729.out" | cut -d= -f2)"
export FILM_FZ_OFF
echo "[0729] FILM_FZ_OFF=$FILM_FZ_OFF"

# ---- 3. four runs in parallel (GPU 5) -------------------------------------------
mkdir -p "$DIR/logs"
film() { # run_name inject mask extra_log
  FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_INJECT="$2" FILM_MASK_FORCE="$3" \
    RUN_NAME="$1" INIT_CKPT="$BASE" \
    DATASET_REPO="$REPO" DATASET_ROOT="$RT" FILM_DATASET_ROOT="$RT" \
    "$DIR/train_film.sh" --policy.input_features="$IF15" --steps=50000 --save_freq=5000 \
    --policy.scheduler_decay_steps=50000 \
    > "$DIR/logs/orch_$4.out" 2>&1
}
BASE="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"

HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$RT" RUN_NAME=smolvla_naive_0729 \
  "$DIR/train_smolvla.sh" --steps=50000 --save_freq=5000 --policy.scheduler_decay_steps=50000 \
  > "$DIR/logs/orch_naive_0729.out" 2>&1 &
P1=$!
film smolvla_film_0729_prefix_mask1 prefix 1 film_0729_pm1 & P2=$!
film smolvla_film_0729_prefix_mask0 prefix 0 film_0729_pm0 & P3=$!
film smolvla_film_0729_suffix_mask1 suffix 1 film_0729_sm1 & P4=$!
echo "[0729] pids naive=$P1 pm1=$P2 pm0=$P3 sm1=$P4"
R1=0; wait $P1 || R1=$?; echo "[0729] naive rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[0729] prefix_mask1 rc=$R2"
R3=0; wait $P3 || R3=$?; echo "[0729] prefix_mask0 rc=$R3"
R4=0; wait $P4 || R4=$?; echo "[0729] suffix_mask1 rc=$R4"

# sanity: no stray relative-path outputs
[[ -d "$HOME/outputs" ]] && echo "[0729] WARNING: stray \$HOME/outputs appeared!" >&2

# ---- 4. per-run: best-checkpoint selection (val) + prune + probe(val) -----------
best() { # run film_env...
  local run="$1"; shift
  env "$@" "$PY" "$DIR/select_best_ckpt.py" --run "$DIR/outputs/$run" \
    --val-root "$RV" --repo-id "$REPO_VAL" --prune >> "$DIR/logs/best_0729.out" 2>&1 \
    && echo "[0729] best($run): $(readlink "$DIR/outputs/$run/checkpoints/best" 2>/dev/null)"
}
probe() { # tag run inject mask [c0 c1]
  env FILM_COND=contact,fz,seal FILM_MASK_FORCE="$4" FILM_INJECT="$3" FILM_FZ_OFF="$FILM_FZ_OFF" \
    "$PY" "$DIR/probe_film_authority.py" --checkpoint "$DIR/outputs/$2/checkpoints/best" \
    --dataset-root "$RV" --repo-id "$REPO_VAL" --contact-n 6 \
    ${5:+--c0 "$5" --c1 "$6"} > "$DIR/probes/0729_$1.txt" 2>&1; echo "[0729] probe $1 rc=$?"
}
best smolvla_naive_0729
best smolvla_film_0729_prefix_mask1 FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"
best smolvla_film_0729_prefix_mask0 FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=0 FILM_FZ_OFF="$FILM_FZ_OFF"
best smolvla_film_0729_suffix_mask1 FILM_COND=contact,fz,seal FILM_INJECT=suffix FILM_MASK_FORCE=1 FILM_FZ_OFF="$FILM_FZ_OFF"
probe pm1_std smolvla_film_0729_prefix_mask1 prefix 1
probe pm1_real smolvla_film_0729_prefix_mask1 prefix 1 "0,0,0" "0.6,0.42,0"
probe pm0_std smolvla_film_0729_prefix_mask0 prefix 0
probe pm0_real smolvla_film_0729_prefix_mask0 prefix 0 "0,0,0" "0.6,0.42,0"
probe sm1_std smolvla_film_0729_suffix_mask1 suffix 1
probe sm1_real smolvla_film_0729_suffix_mask1 suffix 1 "0,0,0" "0.6,0.42,0"
grep -a -H -A2 "verdict" "$DIR"/probes/0729_*.txt | grep -v "^--$"
echo "[0729] DONE naive=$R1 pm1=$R2 pm0=$R3 sm1=$R4"
[[ $R1 == 0 && $R2 == 0 && $R3 == 0 && $R4 == 0 ]]
