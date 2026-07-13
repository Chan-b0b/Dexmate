# Environment

This folder pins the devcontainer's package set so a pull of this branch can
reproduce it elsewhere.

| File | Contents |
|---|---|
| `requirements.txt` | `pip freeze` of `/opt/venv` (245 packages) |
| `apt-packages.txt` | `apt-mark showmanual` (350 packages, mostly already part of the base image below -- kept for reference, not meant to be installed line-by-line) |
| `Dockerfile` | Layers `requirements.txt` on top of a base image you supply |
| `freeze.sh` | Regenerates the two files above from the running container |

## What this can and can't reproduce

The running devcontainer is **Jetson Thor (aarch64)**: Ubuntu 24.04, L4T
R38.4.0, CUDA 13.0, cuDNN 9.20, ROS Humble built for Ubuntu 24.04, PyTorch
2.11.0 for aarch64/CUDA 13 -- all pulled from a Jetson-AI-Lab package mirror
(`PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130`). That OS/driver/ROS
layer is tied to the physical Tegra hardware and isn't something a Dockerfile
can rebuild from a generic Ubuntu or `nvidia/cuda` image -- it has to come
from Dexmate's own Thor base image.

So `environment/Dockerfile` only reproduces the layer on top of that base
image: the Python packages in `requirements.txt`, installed from the same
index. Build it with:

```bash
docker build -f environment/Dockerfile --build-arg BASE_IMAGE=<dexmate-thor-base-image> .
```

If you don't know `<dexmate-thor-base-image>`, ask whoever manages the Thor
provisioning/imaging for this robot fleet -- it isn't discoverable from
inside a running container (no docker socket in here).

## Keeping this in sync

After installing new packages in the devcontainer:

```bash
./environment/freeze.sh
git add environment/requirements.txt environment/apt-packages.txt
git commit -m "environment: refresh package pins"
```
