# Reproduces the Python/ROS package set of the Jetson Thor devcontainer this
# project runs in. It does NOT rebuild the OS/CUDA/L4T layer -- that comes
# from Dexmate's Thor base image and is tied to the physical Tegra hardware
# (L4T R38.4.0, CUDA 13.0, cuDNN 9.20, ROS Humble built for Ubuntu 24.04/aarch64).
# Pass that base image via BASE_IMAGE; see environment/ENVIRONMENT.md.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130 \
    PIP_TRUSTED_HOST=jetson-ai-lab.io \
    HF_HOME=/data/models/huggingface \
    TRANSFORMERS_CACHE=/data/models/huggingface \
    TORCH_HOME=/data/models/torch \
    CUDA_HOME=/usr/local/cuda \
    ROS_DISTRO=humble

COPY environment/requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
