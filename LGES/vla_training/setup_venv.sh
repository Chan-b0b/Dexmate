#!/usr/bin/env bash
# Build the training venv this directory's scripts expect: /home/maverick/vla_venv
# (train_smolvla.sh / train_film.sh honour $VENV; run_case_pick_*.sh and
# extend_50k_0727.sh hardcode that path).
#
#   ./setup_venv.sh              # create/repair the venv + data dirs
#   VENV=/somewhere ./setup_venv.sh
#
# Idempotent: re-running syncs the venv to requirements.lock.txt. Safe while a
# training runs only if the lock hasn't changed — otherwise wait for it to finish.
#
# Host this was validated on: 8x B300 SXM6, driver 580 (CUDA 13.0), Ubuntu 24.04,
# x86_64, Python 3.12.3. See requirements.in for why each version is pinned.
set -euo pipefail

VENV="${VENV:-/home/maverick/vla_venv}"
DATA_ROOT="${DATA_ROOT:-/data001/maverick}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"

# Wheel cache on the data volume too — it grows to ~6 GB and / is the shared root.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_ROOT/cache/uv}"

# uv, not pip: it resolves and installs the ~6 GB CUDA stack in a couple of minutes
# and hardlinks from its cache, so repeated venv builds are nearly free.
if [[ ! -x "$UV" ]]; then
  echo "[setup] installing uv -> $HOME/.local/bin"
  mkdir -p "$HOME/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
fi

# datasets / outputs / logs in this directory are symlinks into $DATA_ROOT; the HF
# cache is symlinked from ~/.cache/huggingface because the scripts export
# HF_HOME="$HOME/.cache/huggingface". Nothing large lives under the repo itself.
mkdir -p "$DATA_ROOT"/vla_training/{datasets,outputs,logs} "$DATA_ROOT/hf_cache"
mkdir -p "$HOME/.cache"
[[ -e "$HOME/.cache/huggingface" ]] || ln -s "$DATA_ROOT/hf_cache" "$HOME/.cache/huggingface"
for p in datasets outputs logs; do
  if [[ -L "$DIR/$p" && ! -d "$DIR/$p/" ]]; then
    echo "[setup] repointing dangling symlink $p -> $DATA_ROOT/vla_training/$p"
    rm "$DIR/$p"; ln -s "$DATA_ROOT/vla_training/$p" "$DIR/$p"
  fi
done

echo "[setup] venv -> $VENV"
# --allow-existing so re-running syncs an existing venv instead of refusing (and
# without --clear, which would wipe one a training is currently using).
"$UV" venv --seed --python 3.12 --allow-existing "$VENV"
# --extra-index-url for the +cu130 wheels; unsafe-best-match lets the resolver take
# torch from download.pytorch.org and everything else from PyPI.
"$UV" pip install --python "$VENV/bin/python" -r "$DIR/requirements.lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

# smolvla_meanflow is a lerobot plugin that lives in this repo, so it is installed
# from source with --no-deps (it declares none on purpose).
if [[ -f "$DIR/smolvla_meanflow/pyproject.toml" ]]; then
  "$UV" pip install --python "$VENV/bin/python" --no-deps -e "$DIR/smolvla_meanflow"
fi

echo "[setup] verifying"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" - <<'EOF'
import torch, lerobot, transformers, accelerate
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  gpus {torch.cuda.device_count()}")
print(f"  {torch.cuda.get_device_name(0)}  cc {torch.cuda.get_device_capability(0)}")
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite((a @ a).float()).all(), "bf16 matmul produced non-finite values"
print(f"  lerobot {lerobot.__version__}  transformers {transformers.__version__}"
      f"  accelerate {accelerate.__version__}  (bf16 matmul OK)")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
EOF

echo "[setup] done."
echo "        hub login (once, for private/gated repos): $VENV/bin/hf auth login"
# There is no system-wide pip on this host (and no root to add one), so point at the
# venv's — a bare `pip install` outside the venv just fails with "No module named pip".
echo "        adding a package: source $VENV/bin/activate && pip install <pkg>"
echo "        (make it permanent: add it to requirements.in and regenerate the lock)"
