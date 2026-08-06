# Training container for the VLA work in LGES/vla_training (SmolVLA / π0.5 / X-VLA /
# FiLM variants, all through lerobot 0.5.1). Target host: 8x NVIDIA B300 SXM6,
# driver 580 (CUDA 13.0), Ubuntu 24.04, x86_64.
#
# The point of this image is that the scripts in this directory run *unmodified*
# inside it: they hardcode `VENV=/home/maverick/vla_venv`, `HF_HOME=$HOME/.cache/
# huggingface`, and reach datasets/outputs/logs through symlinks into
# /data001/maverick/vla_training. All three paths are reproduced here, so
# the container is a drop-in replacement for the venv that used to live on the box.
#
# CUDA 13 base (not NGC): NGC 26.xx ships torch 2.13a0, which is far outside
# lerobot 0.5.1's `torch<2.11` pin. Plain cu130 wheels give us torch 2.10 +
# torchvision 0.25 + torchcodec 0.10 — a triple that satisfies every lerobot pin
# and still carries sm_100 cubins (what NGC itself targets for B300).
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ARG UID=1007
ARG GID=1007
ARG USER=maverick

ARG DEBIAN_FRONTEND=noninteractive
# ffmpeg: torchcodec links the libav* shared libs (video-mode datasets). The
# existing datasets are image-mode — a Jetson-era constraint that does not apply
# here — so video mode is available if a future dataset wants it. torchcodec's
# CUDA core also needs libnppicc.so.13, which it does NOT pull in via pip; it
# comes from the base image's libnpp-13-0. That is a second reason for the CUDA
# base image over plain ubuntu + pip torch.
# git-lfs: hub dataset/checkpoint pulls. rsync/tmux: the storage + long-run workflow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential ca-certificates curl git git-lfs \
        ffmpeg rsync tmux procps less vim-tiny \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# Same uid/gid/name as the host account, so files written to the mounted repo and
# the mounted data volume stay owned by the user outside the container.
RUN groupadd -g ${GID} ${USER} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

ENV VENV=/home/${USER}/vla_venv
ENV DATA_ROOT=/data001/maverick

# HF cache lives on the data volume; the scripts here read $HOME/.cache/huggingface
# (see extend_50k_0727.sh), which move_to_data.sh had symlinked there on the host.
# Reproduce both the symlink and the env var. The HF token is $HF_HOME/token.
RUN mkdir -p /home/${USER}/.cache ${DATA_ROOT} && \
    ln -s ${DATA_ROOT}/hf_cache /home/${USER}/.cache/huggingface && \
    chown -R ${UID}:${GID} /home/${USER} ${DATA_ROOT}

# Build context is LGES/vla_training/ (see docker-compose.yml) — the lock is shared
# with the plain-venv setup in setup_venv.sh.
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN python3.12 -m venv ${VENV} && \
    ${VENV}/bin/pip install --no-cache-dir --upgrade pip && \
    ${VENV}/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /tmp/requirements.lock.txt && \
    rm /tmp/requirements.lock.txt && \
    chown -R ${UID}:${GID} ${VENV}

ENV PATH=${VENV}/bin:${PATH} \
    HF_HOME=${DATA_ROOT}/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    WANDB_DIR=${DATA_ROOT}/wandb \
    TRITON_CACHE_DIR=${DATA_ROOT}/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=${DATA_ROOT}/cache/inductor \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER ${USER}
WORKDIR /home/${USER}/Dexmate/LGES/vla_training
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
