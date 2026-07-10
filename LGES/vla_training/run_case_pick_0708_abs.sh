#!/usr/bin/env bash
# Same 3-run comparison as run_case_pick_0708.sh but on the ABSOLUTE-action
# dataset (Chanho-Lee/lges_case_pick_0708_abs, action 8 = xyz+quat wxyz+suction):
#   1. smolvla_naive_0708_abs — vanilla finetune of lerobot/smolvla_base (10k)
#   2. film_on_naive_0708_abs — FiLM (v2) finetune FROM the naive checkpoint (2k)
#   3. smolvla_film_0708_abs  — FiLM (v2) finetune FROM smolvla_base directly (10k)
# Runs 1 and 3 train CONCURRENTLY on GPU 6; run 2 starts when run 1 finishes.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=6
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0708_abs
ROOT="$DIR/datasets/lges_case_pick_0708_abs"
API="https://huggingface.co/api/datasets/$REPO/tree/main?recursive=true"

RENAME_MAP='{"observation.images.head": "observation.images.camera1", "observation.images.head_depth": "observation.images.camera2"}'
INPUT_FEATURES='{"observation.state": {"type": "STATE", "shape": [15]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]}}'

# ---- 1. wait for the dataset upload to land and settle -----------------------
deadline=$(( $(date +%s) + 12*3600 ))
prev=""
while :; do
  tree="$(curl -s "$API" || true)"
  if echo "$tree" | grep -q '"meta/info.json"'; then
    sig="$(echo "$tree" | md5sum | cut -d' ' -f1)"
    n="$(echo "$tree" | grep -o '"path"' | wc -l)"
    if [[ "$sig" == "$prev" ]]; then
      echo "[orch-abs] dataset upload settled ($n files)"; break
    fi
    echo "[orch-abs] dataset uploading... ($n files, waiting for tree to settle)"
    prev="$sig"
  else
    echo "[orch-abs] dataset repo still empty, polling..."
  fi
  if (( $(date +%s) > deadline )); then
    echo "[orch-abs] TIMEOUT: $REPO still has no meta/info.json after 12h" >&2; exit 1
  fi
  sleep 90
done

# ---- 2. download + schema sanity check ---------------------------------------
"$PY" - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("$REPO", repo_type="dataset", local_dir="$ROOT")
print("[orch-abs] dataset downloaded to", p)
EOF

"$PY" - <<EOF
import json
info = json.load(open("$ROOT/meta/info.json"))
f = info["features"]
st = list(f["observation.state"]["shape"])
act = list(f["action"]["shape"])
cams = sorted(k for k in f if k.startswith("observation.images."))
print(f"[orch-abs] episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
print(f"[orch-abs] state shape={st} action shape={act} cams={cams}")
assert st == [15], f"state shape {st} != [15] — FiLM indices (seal=8, wrench=9:15) would be wrong"
assert "observation.images.head" in cams, "missing observation.images.head"
assert "observation.images.head_depth" in cams, "missing observation.images.head_depth"
json.load(open("$ROOT/meta/stats.json"))["observation.state"]  # FiLM needs these
print("[orch-abs] schema OK")
EOF

# ---- 3. the three runs --------------------------------------------------------
NAIVE=smolvla_naive_0708_abs
mkdir -p "$DIR/logs"

echo "[orch-abs] === launching RUN 1 (naive, $NAIVE) and RUN 3 (smolvla_film_0708_abs) in parallel on GPU 6 ==="

HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$ROOT" RUN_NAME="$NAIVE" \
  "$DIR/train_smolvla.sh" --steps=10000 \
  > "$DIR/logs/orch_naive_abs.out" 2>&1 &
PID_NAIVE=$!

BASE_CKPT="$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"
FILM_VARIANT=v2 RUN_NAME=smolvla_film_0708_abs \
  INIT_CKPT="$BASE_CKPT" \
  DATASET_REPO="$REPO" DATASET_ROOT="$ROOT" FILM_DATASET_ROOT="$ROOT" \
  "$DIR/train_film.sh" \
  --policy.input_features="$INPUT_FEATURES" \
  --steps=10000 \
  > "$DIR/logs/orch_film_base_abs.out" 2>&1 &
PID_FILM_BASE=$!

echo "[orch-abs] naive pid=$PID_NAIVE (log: logs/orch_naive_abs.out)  film_base pid=$PID_FILM_BASE (log: logs/orch_film_base_abs.out)"

NAIVE_RC=0;     wait "$PID_NAIVE"     || NAIVE_RC=$?
echo "[orch-abs] RUN 1 (naive) finished rc=$NAIVE_RC"

FILM_NAIVE_RC=-1
NAIVE_CKPT="$DIR/outputs/$NAIVE/checkpoints/last/pretrained_model"
if [[ "$NAIVE_RC" == "0" && -d "$NAIVE_CKPT" ]]; then
  echo "[orch-abs] === RUN 2: FiLM v2 on top of naive (film_on_naive_0708_abs) ==="
  FILM_NAIVE_RC=0
  FILM_VARIANT=v2 RUN_NAME=film_on_naive_0708_abs \
    INIT_CKPT="$NAIVE_CKPT" \
    DATASET_REPO="$REPO" DATASET_ROOT="$ROOT" FILM_DATASET_ROOT="$ROOT" \
    "$DIR/train_film.sh" --steps=2000 \
    > "$DIR/logs/orch_film_naive_abs.out" 2>&1 || FILM_NAIVE_RC=$?
  echo "[orch-abs] RUN 2 (film_on_naive) finished rc=$FILM_NAIVE_RC"
else
  echo "[orch-abs] RUN 2 SKIPPED: naive run failed (rc=$NAIVE_RC) or checkpoint missing" >&2
fi

FILM_BASE_RC=0; wait "$PID_FILM_BASE" || FILM_BASE_RC=$?
echo "[orch-abs] RUN 3 (film_from_base) finished rc=$FILM_BASE_RC"

echo "[orch-abs] DONE  naive=$NAIVE_RC film_on_naive=$FILM_NAIVE_RC film_from_base=$FILM_BASE_RC"
echo "[orch-abs] outputs: $DIR/outputs/{$NAIVE,film_on_naive_0708_abs,smolvla_film_0708_abs}"
[[ "$NAIVE_RC" == "0" && "$FILM_NAIVE_RC" == "0" && "$FILM_BASE_RC" == "0" ]]
