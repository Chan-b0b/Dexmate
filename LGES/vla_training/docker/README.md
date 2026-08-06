# Training container (optional)

**On this host, use [`../setup_venv.sh`](../setup_venv.sh) instead** — it builds the same
stack from the same [`../requirements.lock.txt`](../requirements.lock.txt) into
`/home/maverick/vla_venv`, and that is what has actually been run and verified here. This
directory exists for portability to a machine where a container is preferred.

Same contents either way — SmolVLA, π0.5, X-VLA, the FiLM variants and `smolvla_meanflow`
through **lerobot 0.5.1**, built for 8× **B300 SXM6** (driver 580 / CUDA 13.0, Ubuntu
24.04, x86_64). The image reproduces the paths the scripts assume: the venv at
`/home/maverick/vla_venv`, `$HOME/.cache/huggingface` symlinked to the data volume, and
`/data001/maverick` as a mount — so `train_smolvla.sh`, `train_film.sh`,
`extend_50k_0727.sh`, `run_case_pick_*.sh` and the `datasets|outputs|logs` symlinks all
work **unmodified**.

Robot-side scripts (`run_policy.py`, `collect_case_pick.py`, `probe_*_live.py`) are
*not* covered — no `dexcontrol`/zenoh here. This image trains and evaluates offline.

## Prerequisites

- `nvidia-container-toolkit` (present on this host) and a driver ≥ 580.
- **Your account must be able to talk to the docker socket.** It currently cannot:
  ```bash
  sudo usermod -aG docker "$USER"   # then log out / back in, or: newgrp docker
  ```
  Everything below assumes that is done (otherwise prefix with `sudo`).

## One-time setup

```bash
cd LGES/vla_training/docker
cp .env.example .env                       # then edit HOST_DATA_ROOT if needed
mkdir -p "$(grep -oP '(?<=^HOST_DATA_ROOT=).*' .env)"   # must be owned by you
docker compose build                       # 10–20 min; ~15 GB image (CUDA devel + a 6 GB venv)
```

The data root holds datasets, checkpoints and the HF cache — budget 100 GB+ (the previous
run used 12 GB datasets + 57 GB outputs + 57 GB HF cache). It defaults to
`/data001/maverick` on this host: a 7 TB NVMe with 6.4 TB free, and the same root the venv
setup uses, so the two share one copy of the datasets and checkpoints.

Verify the GPUs and the stack:

```bash
docker compose run --rm vla python -c "
import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
print((torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16) @
       torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)).float().mean().item())
import lerobot, transformers, torchcodec
print(lerobot.__version__, transformers.__version__, torchcodec.__version__)"
```

Expected: `2.10.0+cu130 13.0 8`, `NVIDIA B300 SXM6 AC (10, 3) [... 'sm_100', 'sm_120', ...]`,
a finite mean, `0.5.1 5.3.0 0.10.0`.

This exact stack is verified on the host in a plain venv (`../setup_venv.sh`), including a
real 2-step SmolVLA run on `Chanho-Lee/lges_case_pick_0729_val` that wrote a checkpoint.
**The image build itself has not been run** — the account this was authored from had no
docker socket access.

Log in to the hub once (the token persists on the data volume, at `$HF_HOME/token`):

```bash
docker compose run --rm vla hf auth login
```

## Running training

Start a long-lived container and work inside it — this matches the existing workflow
(`nohup`/`tmux` + `pgrep`-based orchestration in `extend_50k_0727.sh`):

```bash
docker compose up -d
docker compose exec vla bash
```

Inside, everything is as before:

```bash
RUN_NAME=b300_run_01 ./train_smolvla.sh --steps=60000
FILM_VARIANT=v2 RUN_NAME=film_v2 ./train_film.sh
./run_case_pick_0729.sh          # these pin their own GPU (CUDA_VISIBLE_DEVICES=5 inside)
nohup ./extend_50k_0727.sh &     # the multi-run orchestrator; all 8 GPUs are visible
```

Or fire a single run without a shell:

```bash
docker compose run --rm -e RUN_NAME=b300_run_01 vla ./train_smolvla.sh --steps=60000
```

### One model across all 8 GPUs

lerobot 0.5.1 wraps its loop in `accelerate`, so DDP needs no code change:

```bash
accelerate launch --num_processes=8 --mixed_precision=bf16 \
  "$VENV/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=Chanho-Lee/lges_case_pick_0721_0727 \
  --dataset.root="$PWD/datasets/lges_case_pick_0721_0727" \
  --batch_size=32 --num_workers=8 --steps=60000 --save_freq=2000 --log_freq=50 \
  --output_dir="$PWD/outputs/ddp8" --job_name=ddp8
```

`--batch_size` is **per process** — the run above is an effective batch of 256, so the
lr schedule from the single-GPU runs no longer transfers directly. `--num_workers` is
also per process; 8 × 8 = 64 loader workers is a sane start on 256 cores (8 × 32 is not).

The wrapper scripts (`train_smolvla.sh`, `train_film.sh`) call the venv binary directly
and are therefore single-process; use the `accelerate launch` form above for DDP.

### TensorBoard

`tb_log.py` is started by the wrappers and parses `train.log` into scalars. Port 6006
is published:

```bash
docker compose exec vla tensorboard --logdir logs/b300_run_01/tb --host 0.0.0.0
# → http://<host>:6006
```

## What's pinned, and why

| | version | reason |
|---|---|---|
| lerobot | 0.5.1 | every wrapper here targets it (`train_pi05.py`'s processor shim, `train_xvla.py`'s `florence_config` patch) |
| torch / torchvision / torchcodec | 2.10.0 / 0.25.0 / 0.10.0, all `+cu130` | the only triple that satisfies lerobot 0.5.1's pins (`torch<2.11`, `torchvision<0.26`, `torchcodec<0.11`) *and* has CUDA 13 wheels |
| transformers | 5.3.0 | lerobot's own pin; 5.5 breaks its groot dataclass |
| base image | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` | NGC PyTorch 26.xx ships torch 2.13a0 — far outside lerobot's pin, and no matching torchcodec wheel exists |

B300 is `sm_103`; the cu130 wheels carry `sm_100` cubins plus PTX, which is exactly what
NVIDIA's own NGC images target for this part (`TORCH_CUDA_ARCH_LIST=… 10.0 12.0+PTX`) —
minor-revision-forward binary compatibility covers it.

`pandas` is held at 2.x on purpose (3.0 changes the default string dtype / copy-on-write
and is not validated against these scripts).

Unlike the Jetson venv, **torchcodec works here**, so video-mode LeRobot datasets are an
option. The existing datasets are image-mode — a Thor constraint, not a requirement —
and load fine either way. torchcodec needs `libnppicc.so.13`, which pip does not provide;
it comes from the base image's `libnpp-13-0`. (lerobot imports torchcodec lazily, so even
a broken torchcodec would only affect video-mode datasets.)

### Changing dependencies

Edit `../requirements.in`, regenerate the lock (command is in that file's header), rebuild.
Don't hand-edit `../requirements.lock.txt`. The lock is shared with `../setup_venv.sh`, so
the venv and the image never drift apart. The build context is `..` for that reason.

`smolvla_meanflow` is editable-installed by the entrypoint on container start (it lives
in the mounted repo, so it can't be baked into the image).

## Notes

- `--gpus all` + `ipc: host` are set in the compose file; without host IPC, 32 dataloader
  workers exhaust docker's 64 MB `/dev/shm`.
- The container runs as uid/gid 1007 (`maverick`), so checkpoints and logs written to the
  mounts stay owned by your host account.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `OMP_NUM_THREADS=8` are baked in
  (the former matches `extend_50k_0727.sh`; the latter stops torch from opening 256 threads
  per process on this core count).
