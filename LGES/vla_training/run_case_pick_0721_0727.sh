#!/usr/bin/env bash
# 0721_0727 round (new robot, combined dataset): download + validate force profile
# (rise-based contact, F0=6/tau=4/fz_tau=5 — same calibration as 0721 unless the
# validation says otherwise), derive dF, then train 3 settings in parallel on GPU 7:
#   1. smolvla_naive_0721_0727
#   2. smolvla_film_0721_0727_dF_prefix_mask1      (winning family: 4ch prefix+mask1)
#   3. smolvla_film_0721_0727_dF_prefix_mask1_os3  (+ transition oversampling x3)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=7
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
export FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5

REPO=Chanho-Lee/lges_case_pick_0721_0727
ROOT="$DIR/datasets/lges_case_pick_0721_0727"
ROOT_DF="$DIR/datasets/lges_case_pick_0721_0727_dF"

rm -rf "$ROOT"
"$PY" -c "from huggingface_hub import snapshot_download as s; s('$REPO', repo_type='dataset', local_dir='$ROOT')"

"$PY" - <<EOF | tee "$DIR/logs/validate_0727.out"
import glob, json
import numpy as np, pandas as pd
info = json.load(open("$ROOT/meta/info.json"))
f = info["features"]
st = list(f["observation.state"]["shape"]); act = list(f["action"]["shape"])
cams = sorted(k for k in f if k.startswith("observation.images."))
print(f"[0727] episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
print(f"[0727] state={st} action={act} ({'abs' if act==[8] else 'rel' if act==[7] else '??'}) cams={cams}")
assert st == [15] and "observation.images.head" in cams and "observation.images.head_depth" in cams

hover_F, press_F, contact_z, all_fz = [], [], [], []
for p in sorted(glob.glob("$ROOT/data/*/*.parquet")):
    df = pd.read_parquet(p, columns=["observation.state", "episode_index"])
    stt = np.stack(df["observation.state"].to_numpy())
    all_fz.append(stt[:, 11])
    for ep in df["episode_index"].unique():
        s = stt[(df["episode_index"] == ep).to_numpy()]
        fmag = np.linalg.norm(s[:, 9:12], axis=1)
        hover_F.append(np.median(fmag[:15]))
        seal = np.flatnonzero(s[:, 8] > 0.5)
        if len(seal): press_F.append(np.median(fmag[seal]))
        jumps = np.flatnonzero(np.abs(np.diff(fmag, prepend=fmag[0])) >= 2)
        if len(jumps): contact_z.append(s[jumps[0], 2])
hF, pF = np.median(hover_F), np.median(press_F)
cz = np.array(contact_z)
fzm = float(np.median(np.concatenate(all_fz)))
print(f"[0727] |F| hover={hF:.1f}N sealed={pF:.1f}N ({'RISE' if pF>hF else 'DROP'})")
print(f"[0727] contact z p10/50/90={np.percentile(cz,[10,50,90]).round(3)} (n={len(cz)})")
print(f"[0727] FZ_MEDIAN={fzm:.1f}")
assert pF > hF, "contact is a DROP -- code is rise-based"
assert hF < 6.0 < pF, f"F0=6 not between hover({hF:.1f}) and press({pF:.1f}) -- recalibrate"
print("[0727] force profile OK for F0=6/tau=4")
EOF
[[ ${PIPESTATUS[0]} == 0 ]] || { echo "[0727] VALIDATION FAILED -- not training" >&2; exit 1; }
# fz centering offset = dataset fz median ("기왕 하는김에 fz 중앙값 맞추자", 2026-07-28).
# The 0721 generation used the legacy fixed -20; this round bakes the measured median.
FILM_FZ_OFF="$(grep -o 'FZ_MEDIAN=[0-9.-]*' "$DIR/logs/validate_0727.out" | cut -d= -f2)"
[[ -n "$FILM_FZ_OFF" ]] || { echo "[0727] could not parse FZ_MEDIAN" >&2; exit 1; }
export FILM_FZ_OFF
echo "[0727] using FILM_FZ_OFF=$FILM_FZ_OFF"

"$PY" "$DIR/derive_df_dataset.py" "$ROOT" "$ROOT_DF"
[[ $? == 0 ]] || { echo "[0727] dF derivation FAILED" >&2; exit 1; }

IF15='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
IF16='{"observation.state": {"type": "STATE", "shape": [16]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'
BASE="$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerobot/smolvla_base'))")"
mkdir -p "$DIR/logs"

HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$ROOT" RUN_NAME=smolvla_naive_0721_0727 \
  "$DIR/train_smolvla.sh" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_naive_0727.out" 2>&1 &
P1=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  RUN_NAME=smolvla_film_0721_0727_dF_prefix_mask1 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0721_0727_dF DATASET_ROOT="$ROOT_DF" FILM_DATASET_ROOT="$ROOT_DF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0727_dF.out" 2>&1 &
P2=$!

FILM_VARIANT=v2 FILM_COND=contact,fz,seal,dfmag FILM_INJECT=prefix FILM_MASK_FORCE=1 \
  FILM_OVERSAMPLE_BOOST=3 \
  RUN_NAME=smolvla_film_0721_0727_dF_prefix_mask1_os3 INIT_CKPT="$BASE" \
  DATASET_REPO=Chanho-Lee/lges_case_pick_0721_0727_dF DATASET_ROOT="$ROOT_DF" FILM_DATASET_ROOT="$ROOT_DF" \
  "$DIR/train_film.sh" --policy.input_features="$IF16" --steps=30000 --save_freq=4000 \
  > "$DIR/logs/orch_film_0727_dF_os3.out" 2>&1 &
P3=$!

echo "[0727] pids naive=$P1 film=$P2 film_os3=$P3"
R1=0; wait $P1 || R1=$?; echo "[0727] naive rc=$R1"
R2=0; wait $P2 || R2=$?; echo "[0727] film rc=$R2"
R3=0; wait $P3 || R3=$?; echo "[0727] film_os3 rc=$R3"
echo "[0727] DONE naive=$R1 film=$R2 film_os3=$R3"
[[ $R1 == 0 && $R2 == 0 && $R3 == 0 ]]
