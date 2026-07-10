#!/usr/bin/env bash
# Orchestrate the 0708 case_pick comparison on the H200 server (GPU 6):
#   1. smolvla_naive_0708   — vanilla finetune of lerobot/smolvla_base
#   2. film_on_naive_0708   — FiLM (v2) finetune FROM the naive checkpoint (20k)
#   3. smolvla_film_0708    — FiLM (v2) finetune FROM smolvla_base directly (60k,
#                             comparable to the naive run)
# Runs 1 and 3 train CONCURRENTLY on GPU 6; run 2 starts when run 1 finishes.
# Dataset: HF Chanho-Lee/lges_case_pick_0708 (same convert_to_lerobot.py schema:
# state 15 = pos3+quat4+suction+seal+wrench6, cameras head + head_depth).
# Waits for the repo upload to finish (tree stable + meta/info.json) before starting.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VENV=/home/maverick/vla_venv
PY="$VENV/bin/python"
export CUDA_VISIBLE_DEVICES=6
# The shared /data/cache/hf has an unreadable token file — use a per-user cache.
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true

REPO=Chanho-Lee/lges_case_pick_0708
ROOT="$DIR/datasets/lges_case_pick_0708"
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
      echo "[orch] dataset upload settled ($n files)"; break
    fi
    echo "[orch] dataset uploading... ($n files, waiting for tree to settle)"
    prev="$sig"
  else
    echo "[orch] dataset repo still empty, polling..."
  fi
  if (( $(date +%s) > deadline )); then
    echo "[orch] TIMEOUT: $REPO still has no meta/info.json after 12h" >&2; exit 1
  fi
  sleep 90
done

# ---- 2. download + schema sanity check ---------------------------------------
"$PY" - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("$REPO", repo_type="dataset", local_dir="$ROOT")
print("[orch] dataset downloaded to", p)
EOF

"$PY" - <<EOF
import json
info = json.load(open("$ROOT/meta/info.json"))
f = info["features"]
st = list(f["observation.state"]["shape"])
cams = sorted(k for k in f if k.startswith("observation.images."))
print(f"[orch] episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
print(f"[orch] state shape={st} cams={cams}")
assert st == [15], f"state shape {st} != [15] — FiLM indices (seal=8, wrench=9:15) would be wrong"
assert "observation.images.head" in cams, "missing observation.images.head"
assert "observation.images.head_depth" in cams, "missing observation.images.head_depth"
json.load(open("$ROOT/meta/stats.json"))["observation.state"]  # FiLM needs these
print("[orch] schema OK")
EOF

# ---- 3. the three runs --------------------------------------------------------
# Runs 1 (naive) and 3 (FiLM from base) train concurrently on GPU 6; run 2
# (FiLM from the naive checkpoint) starts as soon as run 1 finishes.
NAIVE=smolvla_naive_0708
mkdir -p "$DIR/logs"

echo "[orch] === launching RUN 1 (naive, $NAIVE) and RUN 3 (smolvla_film_0708) in parallel on GPU 6 ==="

HF_DATASET_REPO="$REPO" HF_CACHE_DIR="$ROOT" RUN_NAME="$NAIVE" \
  "$DIR/train_smolvla.sh" --steps=10000 \
  > "$DIR/logs/orch_naive.out" 2>&1 &
PID_NAIVE=$!

BASE_CKPT="$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))")"
FILM_VARIANT=v2 RUN_NAME=smolvla_film_0708 \
  INIT_CKPT="$BASE_CKPT" \
  DATASET_REPO="$REPO" DATASET_ROOT="$ROOT" FILM_DATASET_ROOT="$ROOT" \
  "$DIR/train_film.sh" \
  --policy.input_features="$INPUT_FEATURES" \
  --steps=10000 \
  > "$DIR/logs/orch_film_base.out" 2>&1 &
PID_FILM_BASE=$!

echo "[orch] naive pid=$PID_NAIVE (log: logs/orch_naive.out)  film_base pid=$PID_FILM_BASE (log: logs/orch_film_base.out)"

NAIVE_RC=0;     wait "$PID_NAIVE"     || NAIVE_RC=$?
echo "[orch] RUN 1 (naive) finished rc=$NAIVE_RC"

FILM_NAIVE_RC=-1
NAIVE_CKPT="$DIR/outputs/$NAIVE/checkpoints/last/pretrained_model"
if [[ "$NAIVE_RC" == "0" && -d "$NAIVE_CKPT" ]]; then
  echo "[orch] === RUN 2: FiLM v2 on top of naive (film_on_naive_0708) ==="
  FILM_NAIVE_RC=0
  FILM_VARIANT=v2 RUN_NAME=film_on_naive_0708 \
    INIT_CKPT="$NAIVE_CKPT" \
    DATASET_REPO="$REPO" DATASET_ROOT="$ROOT" FILM_DATASET_ROOT="$ROOT" \
    "$DIR/train_film.sh" --steps=2000 \
    > "$DIR/logs/orch_film_naive.out" 2>&1 || FILM_NAIVE_RC=$?
  echo "[orch] RUN 2 (film_on_naive) finished rc=$FILM_NAIVE_RC"
else
  echo "[orch] RUN 2 SKIPPED: naive run failed (rc=$NAIVE_RC) or checkpoint missing" >&2
fi

FILM_BASE_RC=0; wait "$PID_FILM_BASE" || FILM_BASE_RC=$?
echo "[orch] RUN 3 (film_from_base) finished rc=$FILM_BASE_RC"

echo "[orch] DONE  naive=$NAIVE_RC film_on_naive=$FILM_NAIVE_RC film_from_base=$FILM_BASE_RC"
echo "[orch] outputs: $DIR/outputs/{$NAIVE,film_on_naive_0708,smolvla_film_0708}"
[[ "$NAIVE_RC" == "0" && "$FILM_NAIVE_RC" == "0" && "$FILM_BASE_RC" == "0" ]]
