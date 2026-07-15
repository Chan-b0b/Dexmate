#!/usr/bin/env bash
# Pretrain the reversed-curiosity ICM (forward-model ensemble) on the 0708
# delta-action case_pick demos, for the RL fine-tuning stage. Three feature
# variants: proprio / vision / both (the feature-space ablation).
#
# Stage 1 extracts a one-time feature cache (frozen vision tower from the
# trained meanflow checkpoint, mean-pooled embed_image tokens -> 960-d/cam).
# Stage 2 trains the three ensembles from that cache — GPU only needed for
# stage 1; stage 2 is minutes even on CPU.
#
# Default GPU 6 (override with GPU=<idx>). Mirrors run_case_pick_0708_*.sh
# conventions: logs/<run>/train.log, per-user HF cache.
#
#   ./run_icm_0708.sh
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=/home/maverick/vla_venv
PLUGIN=/home/maverick/smolvla_meanflow
export CUDA_VISIBLE_DEVICES="${GPU:-6}"
export HF_HOME="$HOME/.cache/huggingface"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
export TOKENIZERS_PARALLELISM=false

REPO="Chanho-Lee/lges_case_pick_0708"
ROOT="$DIR/datasets/lges_case_pick_0708"
CKPT="$DIR/outputs/smolvla_meanflow_0708/checkpoints/last/pretrained_model"
CACHE="$HOME/icm_cache/icm_features_0708.npz"
OUTDIR="$HOME/checkpoints"
LOGDIR="$DIR/logs/icm_0708"
mkdir -p "$LOGDIR" "$HOME/icm_cache"

if [[ ! -e "$CACHE" ]]; then
  echo "[icm] extracting feature cache -> $CACHE (GPU $CUDA_VISIBLE_DEVICES)"
  "$VENV/bin/python" "$PLUGIN/scripts/extract_icm_features.py" \
    --checkpoint "$CKPT" --dataset-root "$ROOT" --repo-id "$REPO" \
    --batch-size 128 --out "$CACHE" 2>&1 | tee "$LOGDIR/extract.log" || exit 1
else
  echo "[icm] cache exists: $CACHE (delete to re-extract)"
fi

for VARIANT in proprio vision both; do
  OUT="$OUTDIR/icm_0708_$VARIANT.pt"
  if [[ -e "$OUT" ]]; then
    echo "[icm] SKIP $VARIANT: $OUT already exists" >&2
    continue
  fi
  echo "[icm] training variant=$VARIANT -> $OUT"
  "$VENV/bin/python" "$PLUGIN/scripts/train_icm.py" \
    --features "$CACHE" --variant "$VARIANT" --out "$OUT" \
    2>&1 | tee "$LOGDIR/train_$VARIANT.log" || exit 1
done

echo "[icm] done. checkpoints:"
ls -la "$OUTDIR"/icm_0708_*.pt

# PUSH=1 uploads all variants to the (private) HF repo Chanho-Lee/icm_case_pick_0708
if [[ "${PUSH:-0}" == "1" ]]; then
  "$VENV/bin/python" - <<'PYEOF'
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
repo_id = f"{api.whoami()['name']}/icm_case_pick_0708"
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
for f in sorted(Path.home().glob("checkpoints/icm_0708_*.pt")):
    print(f"[icm] uploading {f.name} -> {repo_id}")
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id,
                    commit_message=f"upload {f.name}")
print(f"[icm] https://huggingface.co/{repo_id}")
PYEOF
fi
