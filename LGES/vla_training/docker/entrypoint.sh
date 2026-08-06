#!/usr/bin/env bash
# Prepares the mounted volumes so the training scripts find the paths they expect,
# then hands off to the command. Everything here is idempotent and cheap — no torch
# import, so `docker compose run` stays fast.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data001/maverick}"
REPO="${REPO:-/home/maverick/Dexmate}"

# The three symlinks in vla_training/ (datasets, outputs, logs) point into
# $DATA_ROOT/vla_training and are dangling until that volume is mounted.
if [[ ! -w "$DATA_ROOT" ]]; then
  echo "[entrypoint] $DATA_ROOT is not writable by uid $(id -u)." >&2
  echo "             Create the host dir owned by your account before starting:" >&2
  echo "               mkdir -p <host-data-root> && chown $(id -u):$(id -g) <host-data-root>" >&2
  exit 1
fi
mkdir -p "$DATA_ROOT"/{hf_cache,wandb} \
         "$DATA_ROOT"/vla_training/{datasets,outputs,logs} \
         "$DATA_ROOT"/cache/{triton,inductor}

if [[ ! -d "$REPO/LGES/vla_training" ]]; then
  echo "[entrypoint] repo not mounted at $REPO — mount it there so the scripts'" >&2
  echo "             absolute paths resolve (see docker/README.md)." >&2
  exit 1
fi

# smolvla_meanflow is a lerobot plugin living in the repo, so it can't be baked into
# the image (the source arrives via the mount). Editable-install it on first start;
# the .pth lands in the image's venv layer, so this re-runs on a fresh container.
if [[ -f "$REPO/LGES/vla_training/smolvla_meanflow/pyproject.toml" ]] && \
   ! python -c "import smolvla_meanflow" 2>/dev/null; then
  echo "[entrypoint] installing smolvla_meanflow (editable, --no-deps)"
  pip install --no-deps --no-cache-dir -q -e "$REPO/LGES/vla_training/smolvla_meanflow"
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "[entrypoint] GPUs visible: $(nvidia-smi -L | wc -l)  ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
else
  echo "[entrypoint] WARNING: no GPU visible — run with --gpus all / the compose file." >&2
fi

if [[ ! -f "${HF_HOME:-}/token" ]]; then
  echo "[entrypoint] note: no HF token at \$HF_HOME/token — private datasets/checkpoints"
  echo "             will 401. Run 'hf auth login' once; it persists on the data volume."
fi

exec "$@"
