#!/usr/bin/env bash
# Storage relocation (user request 2026-07-28): datasets + model weights (+ HF cache)
# live on the /data volume from now on. Target was /data/home/maverick but it is
# root-owned and maverick has no sudo -> using maverick-owned /data/home/maverick_data
# (rename to /data/home/maverick later once an admin chowns it; only the three
# symlinks below need re-pointing).
# Flow: rsync -> verify sizes -> atomic symlink swap -> sanity checks -> delete
# originals -> relaunch all trainings (extend_50k_0727.sh + fresh ACT).
set -euo pipefail
SRC=/home/maverick/Dexmate/LGES/vla_training
DST=/data/home/maverick_data
HFC=/home/maverick/.cache/huggingface

mkdir -p "$DST/vla_training" "$DST/hf_cache"
echo "[move] rsync datasets (12G)..."
rsync -a "$SRC/datasets/" "$DST/vla_training/datasets/"
echo "[move] rsync outputs (57G)..."
rsync -a "$SRC/outputs/" "$DST/vla_training/outputs/"
echo "[move] rsync hf cache (57G)..."
rsync -a "$HFC/" "$DST/hf_cache/"

for pair in "$SRC/datasets|$DST/vla_training/datasets" "$SRC/outputs|$DST/vla_training/outputs" "$HFC|$DST/hf_cache"; do
  a="${pair%|*}"; b="${pair#*|}"
  sa=$(du -sb "$a" | cut -f1); sb=$(du -sb "$b" | cut -f1)
  # allow tiny drift (metadata); fail if >1% off
  if (( sb < sa - sa/100 )); then echo "[move] SIZE MISMATCH $a=$sa vs $b=$sb" >&2; exit 1; fi
  echo "[move] verified $b ($sb bytes)"
done

mv "$SRC/datasets" "$SRC/datasets.old";  ln -s "$DST/vla_training/datasets" "$SRC/datasets"
mv "$SRC/outputs"  "$SRC/outputs.old";   ln -s "$DST/vla_training/outputs"  "$SRC/outputs"
mv "$HFC" "$HFC.old";                    ln -s "$DST/hf_cache" "$HFC"
echo "[move] symlinks swapped"

# sanity: HF auth via symlinked token + dataset loads through the symlink
/home/maverick/vla_venv/bin/python - <<'EOF'
from huggingface_hub import whoami
print("[move] HF auth:", whoami()["name"])
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("Chanho-Lee/lges_case_pick_0721_0727",
                    root="/home/maverick/Dexmate/LGES/vla_training/datasets/lges_case_pick_0721_0727")
x = ds[10]; assert x["observation.state"].shape[0] == 15
print("[move] dataset loads through symlink OK")
EOF

rm -rf "$SRC/datasets.old" "$SRC/outputs.old" "$HFC.old"
echo "[move] originals deleted; freed on /home:"; df -h /home/maverick | tail -1

# ---- relaunch trainings ---------------------------------------------------------
# ACT lost its progress (killed before first save) -> fresh 50k on GPU 6
export CUDA_VISIBLE_DEVICES=6 HF_HOME="$HFC"
unset HF_DATASETS_CACHE TRANSFORMERS_CACHE || true
rm -rf "$SRC/outputs/act_0721_0727" "$SRC/logs/act_0721_0727"
/home/maverick/vla_venv/bin/lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=Chanho-Lee/lges_case_pick_0721_0727 --dataset.root="$SRC/datasets/lges_case_pick_0721_0727" \
  --batch_size=32 --steps=50000 --save_freq=10000 --log_freq=100 --num_workers=16 \
  --output_dir="$SRC/outputs/act_0721_0727" --job_name=act_0721_0727 \
  > "$SRC/logs/orch_act_0727.out" 2>&1 &
echo "[move] ACT relaunched (fresh, pid $!)"

# everything else resumes from its last checkpoint via the 50k orchestrator
nohup "$SRC/extend_50k_0727.sh" > "$SRC/logs/extend50k_relaunch.out" 2>&1 &
echo "[move] extend_50k orchestrator relaunched (pid $!)"
echo "[move] DONE"
