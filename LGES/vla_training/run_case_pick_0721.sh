#!/usr/bin/env bash
# 0721 round — NEW ROBOT (force dist shift: contact = |F| RISE; fz baseline ~20N;
# film_contact.py already flipped by the user). Dataset has ONLY layers 1 & 5
# (bimodal contact z) — interpolation to layers 2-4 is the robot-eval criterion.
# Waits for the HF upload, validates schema + the new force profile against the
# flipped contact channel (aborts loudly if direction/F0 look wrong), derives the
# dF dataset, then trains 4 settings in parallel on GPU 7:
#   1. smolvla_naive_0721                     (baseline)
#   2. smolvla_film_0721_prefix_mask1         (best 0708 setting)
#   3. smolvla_film_0721_dF_prefix_mask1      (+dfmag — baseline-robust channel)
#   4. smolvla_film_0721_dF_prefix_mask1_os10 (+|d|F||>=2 transition oversampling)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=7
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0721
ROOT="$DIR/datasets/lges_case_pick_0721"
ROOT_DF="$DIR/datasets/lges_case_pick_0721_dF"
API="https://huggingface.co/api/datasets/$REPO/tree/main?recursive=true"

# ---- 1. wait for the upload to land and settle --------------------------------
deadline=$(( $(date +%s) + 48*3600 ))
prev=""
while :; do
  tree="$(curl -s "$API" || true)"
  if echo "$tree" | grep -q '"meta/info.json"'; then
    sig="$(echo "$tree" | md5sum | cut -d' ' -f1)"
    if [[ "$sig" == "$prev" ]]; then echo "[0721] upload settled"; break; fi
    echo "[0721] uploading... (waiting for tree to settle)"; prev="$sig"
  else
    echo "[0721] repo still empty, polling..."
  fi
  (( $(date +%s) > deadline )) && { echo "[0721] TIMEOUT after 48h" >&2; exit 1; }
  sleep 120
done

# ---- 2. download + schema + NEW-ROBOT force profile validation ----------------
rm -rf "$ROOT"
"$PY" -c "from huggingface_hub import snapshot_download as s; s('$REPO', repo_type='dataset', local_dir='$ROOT')"

"$PY" - <<EOF
import glob, json
import numpy as np, pandas as pd
info = json.load(open("$ROOT/meta/info.json"))
f = info["features"]
st_shape = list(f["observation.state"]["shape"]); act = list(f["action"]["shape"])
cams = sorted(k for k in f if k.startswith("observation.images."))
print(f"[0721] episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
print(f"[0721] state={st_shape} action={act} ({'abs' if act==[8] else 'rel' if act==[7] else '??'}) cams={cams}")
assert st_shape == [15], f"state {st_shape} != [15]"
assert "observation.images.head" in cams and "observation.images.head_depth" in cams

hover_F, press_F, hover_fz, contact_z = [], [], [], []
for p in sorted(glob.glob("$ROOT/data/*/*.parquet")):
    df = pd.read_parquet(p, columns=["observation.state", "episode_index"])
    st = np.stack(df["observation.state"].to_numpy())
    for ep in df["episode_index"].unique():
        s = st[(df["episode_index"] == ep).to_numpy()]
        fmag = np.linalg.norm(s[:, 9:12], axis=1)
        hover_F.append(np.median(fmag[:15])); hover_fz.append(np.median(s[:15, 11]))
        seal = np.flatnonzero(s[:, 8] > 0.5)
        if len(seal): press_F.append(np.median(fmag[seal]))
        jumps = np.flatnonzero(np.abs(np.diff(fmag, prepend=fmag[0])) >= 2)
        if len(jumps): contact_z.append(s[jumps[0], 2])
hF, pF = np.median(hover_F), np.median(press_F)
print(f"[0721] |F| hover median={hF:.1f}N  sealed/press median={pF:.1f}N  -> contact = "
      f"{'RISE' if pF > hF else 'DROP'}")
print(f"[0721] fz hover median={np.median(hover_fz):.1f}N (fz channel offset is -20)")
cz = np.array(contact_z)
print(f"[0721] contact z: n={len(cz)} p10/50/90={np.percentile(cz,[10,50,90]).round(3)} "
      f"(expect bimodal: layers 1 & 5)")
assert pF > hF, f"contact is a DROP here (hover {hF:.1f} -> press {pF:.1f}N) but film_contact.py is RISE-based now"
assert hF < 12.0 < pF, f"F0=12 not between hover({hF:.1f}) and press({pF:.1f}) medians -- recalibrate FILM_F0"
print("[0721] force profile OK for rise-based contact, F0=12")
EOF
# set -e is off (the wait-block needs it off) -> abort explicitly on failed validation
[[ $? == 0 ]] || { echo "[0721] VALIDATION FAILED -- not training" >&2; exit 1; }

# ---- 3. derive the dF dataset --------------------------------------------------
"$PY" "$DIR/derive_df_dataset.py" "$ROOT" "$ROOT_DF"

# ---- 4. four runs in parallel on GPU 7 -----------------------------------------
IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
IF16='{"observation.state": {"type": "STATE", "shape": [16]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"
mkdir -p "$DIR/logs"

HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$ROOT" RUN_NAME=smolvla_naive_0721 \
  "$DIR/train_smolvla.sh" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_naive_0721.out" 2>&1 &
P1=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0721_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO="$REPO" DATASET_ROOT="$ROOT" FILM_DATASET_ROOT="$ROOT" \
  "$DIR/train_film.sh" --policy.input_features="$IF15" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0721_prefix_mask1.out" 2>&1 &
P2=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0721_dF_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0721_dF DATASET_ROOT="$ROOT_DF" FILM_DATASET_ROOT="$ROOT_DF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0721_dF.out" 2>&1 &
P3=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  FILM_OVERSAMPLE_BOOST=10 \
  RUN_NAME=smolvla_film_0721_dF_prefix_mask1_os10 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0721_dF DATASET_ROOT="$ROOT_DF" FILM_DATASET_ROOT="$ROOT_DF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0721_dF_os10.out" 2>&1 &
P4=$!

echo "[0721] pids naive=$P1 film=$P2 dF=$P3 dF_os10=$P4"
R1=0; wait $P1 || R1=$?; echo "[0721] naive rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[0721] film_prefix_mask1 rc=$R2"
R3=0; wait $P3 || R3=$?; echo "[0721] film_dF rc=$R3"
R4=0; wait $P4 || R4=$?; echo "[0721] film_dF_os10 rc=$R4"
echo "[0721] DONE naive=$R1 film=$R2 dF=$R3 dF_os10=$R4"
[[ $R1 == 0 && $R2 == 0 && $R3 == 0 && $R4 == 0 ]]
