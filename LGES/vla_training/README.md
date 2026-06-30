# VLA training for the case+battery suction demo

Fine-tunes **SmolVLA (450M)** on takes recorded by `case_battery_demo`
(`run_demo --record`, auto-cut per sub-task). Scope: the six suction-only
left-arm tasks — `case_pick`, `case_place`, `battery_{1,2}_pick`,
`battery_{1,2}_place`. `hand_off` / `gripper_battery_handling` (right arm +
Robotiq) are excluded for now.

## Environment

Everything runs in `/home/dexmate/vla_venv` — an isolated venv that reuses
the working Jetson torch 2.11 / torchvision 0.26 / CUDA from `/opt/venv`
via a `.pth` fallback (`_optvenv_fallback.pth`). `/opt/venv` is never
modified. Installed on top, deliberately ignoring lerobot's pins
(`--no-deps`): `lerobot 0.5.1`, `transformers 5.3.0` (lerobot's pinned
version — 5.5 breaks lerobot's groot dataclass), `datasets`, `av`, etc.

Known platform constraints (Jetson Thor, aarch64):
- **No torchcodec**, and lerobot's pyav fallback needs
  `torchvision.io.VideoReader` (removed in tv 0.26) → datasets are stored
  in **image mode**, never video mode.
- Don't reinstall torch/torchvision in this venv; PyPI aarch64 wheels
  don't cover Thor's GPU.

## Pipeline

```
LGES/recordings/<task>/<take>/        # raw takes (recorder.py)
  └─ convert_to_lerobot.py            # → LeRobotDataset (image mode)
       datasets/lges_suction          #    train split
       datasets/lges_suction_val      #    last 2 takes per task held out
  └─ train_smolvla.sh                 # lerobot-train wrapper
       outputs/<run>/checkpoints/...
```

### 1. Convert

```bash
/home/dexmate/vla_venv/bin/python convert_to_lerobot.py
```

Per-frame features (see the module docstring for full details):
- `observation.images.head` — head RGB, 512x320
- `observation.state` (14) — left EE pos+quat, suction, **raw** 6-axis wrench
  (raw, not tared: the tare baseline is script-side hidden state the policy
  can't reproduce at deployment, and it's orientation-dependent anyway)
- `action` (7) — next-state EE delta (Δpos + rotvec, base frame) +
  next suction command. Observed deltas, not commanded: the 50 Hz command
  trace lived in `/tmp` and was lost to a reboot.
- `task` — per-take instruction from `meta.json`

Only `success: true` takes are used. Re-running deletes and rebuilds the
output datasets.

### 2. Train

```bash
./train_smolvla.sh                      # defaults: bs 32, 20k steps
./train_smolvla.sh --steps=40000        # any lerobot-train override
RUN_NAME=myrun ./train_smolvla.sh       # name the output dir
```

The wrapper maps our single `head` camera onto the pretrained model's
`camera1` slot via `--rename_map` (the base checkpoint expects
`camera1..3`; absent cameras are attention-masked out, which the base
model was trained with).

TensorBoard: lerobot only logs to wandb, so the script tees stdout to
`outputs/<run>/train.log` and `tb_log.py` parses it into scalars in the
background. View with `tensorboard --logdir outputs/<run>/tb` (loss, grad
norm, lr, epoch).

First run downloads `lerobot/smolvla_base` (~1 GB) from the HF hub
(already cached on this machine).
Checkpoints land in `outputs/<run>/checkpoints/`. Don't train while the
robot demo runs — same GPU.

### 3. Evaluate (offline, no robot)

```bash
/home/dexmate/vla_venv/bin/python eval_offline.py   # latest checkpoint vs val set
```

Open-loop per-step action error per task. First 20k-step run:
**1.77 mm position, 3.06 mrad rotation, 96.5% suction** — healthy fit (errors
are ~30–50% of the per-step motion std on the moving axes). Open-loop error
does NOT guarantee closed-loop success; only a guarded on-robot run does.

### 4. On-robot executor

[`run_policy.py`](run_policy.py) — scope is one task (`case_pick`), left arm,
suction (see [EXECUTOR_DESIGN.md](EXECUTOR_DESIGN.md)). Three modes:

```bash
# offline, no robot: validate obs path + rotation convention + predictions
/home/dexmate/vla_venv/bin/python run_policy.py --self-test ../recordings/case_pick/<take>

# live, needs robot, COMMANDS NOTHING: homes, optionally moves to a start
# pose, then prints predicted action / clamped target / box / IK each tick
/home/dexmate/vla_venv/bin/python run_policy.py --dry-run \
    --goto-start ../recordings/case_pick/<take>

# live, COMMANDS THE ARM (guarded). Single task (default case_pick):
/home/dexmate/vla_venv/bin/python run_policy.py --go \
    --goto-start ../recordings/case_pick/<take> --force-limit 15

# chain sub-tasks in sequence — extend the list one at a time. --pause-between
# stops for ENTER before each sub-task so you can verify every handoff.
/home/dexmate/vla_venv/bin/python run_policy.py --go --pause-between \
    --chain case_pick case_place \
    --goto-start ../recordings/case_pick/<take> --force-limit 15
```

Each task stops on its done-signal (**pick = seal then lift back to hover;
place = release then retract to hover** — the recorded episodes include the
lift, and finishing at hover is what puts the next task in-distribution), a
safety abort, ENTER/Ctrl-C, or the per-task `--max-ticks` cap (a task that
hits the cap without finishing is treated as a stall and stops the chain).
Between tasks the policy is reset, the instruction + workspace box switch, and
the **force baseline is re-taken** (so carrying a case/battery doesn't bias the
contact guard). On a clean finish suction is left as-is (a final pick stays
held); any abort/stall drops it.

Startup always homes via `go_to_default_pose` (like run_demo; `--no-home` to
skip) so the arm begins in the joint config the recordings started from.

**`--go` safety layer** (abort = stop + suction off):
- requires `--goto-start` (dry-run proved home is ~70 cm outside the trained box)
- per-step clamp: `dpos ≤ [10,50,50] mm`, `drot ≤ 25 mrad`. Must not be set
  too tight — at half this the arm just hovered (real deltas got clipped and
  never reached the case). These bound spikes while clearing a full pick.
- per-task workspace box hard-stop (each task's recorded EE range + 5 cm; see
  the `TASKS` table in run_policy.py)
- force abort relative to the baseline, **re-taken at each task start** (raw
  wrench has a ~14 N gravity offset + payload weight once holding; `--force-limit`)
- IK-failure: hold the tick, abort after 3 consecutive failures
- operator abort: ENTER any time (or `--pause-between`'s per-task prompt), Ctrl-C,
  plus a per-task `--max-ticks` cap; e-stop always
- an explicit ENTER-to-authorize prompt before the first command

Self-test verdicts (validated offline):
- **FK/obs path** reproduces the recorded EE *exactly* on static frames
  (0 µm). The ObsBuilder reuses the publisher's `_EEKinematics` — the same FK
  that generated the training data.
- **Delta-integration convention** matches the converter to 1e-16 (base-frame
  left-multiply of the rotvec).

## Known data quirk (not a bug to fix urgently)

The recorder samples the joint dict and the EE-FK pose from two *separate*
joint reads within one 15 Hz tick, so during motion the logged joints and
logged EE are a few ms apart (≤0.8 mm EE skew on fast frames, 0 when static).
Harmless for training/deploy (both use the FK value, and per-step motions are
mm-scale), but if the recorder is revisited, read joints once per tick and
derive both from it.

## Not built yet

- Closed-loop validation of `--go` on the real robot (clear-workspace burst
  first, then with the case). Tune clamps / force limit from what's observed.
- Success/termination auto-stop (vacuum seal) — currently manual via max-ticks.
- Recorder fix: copy `/tmp/cns_trace.csv` rows into each take dir at save
  time so commanded actions survive reboots (we lost them to one).
