#!/usr/bin/env python3
"""On-robot executor for the PREV-ACTION SmolVLA model (contact_aware_vla).

Thin wrapper around LGES/vla_training/run_policy.py — it does NOT modify run_policy
and reuses ALL of its machinery (homing, goto-start, IK/Mover, per-step clamps,
force limit, workspace box, IK-fail abort, vacuum-seal stop, retreat-to-hover,
chaining). The ONLY change is the observation: the prev-action model expects a
22-dim state = the usual 15-dim (pos3 quat4 suction1 sealed1 wrench6) PLUS the
previous action (7). We monkey-patch run_policy.predict to build that 22-dim obs
and track the previous action across ticks.

Previous action == the realized EE delta between consecutive observations, built
exactly like convert_prevaction.py made the training state:
    prev = ( pos[t]-pos[t-1],
             rotvec(q[t] * q[t-1]^-1)  (base frame, wxyz),
             suction[t] )            # current commanded suction
and zeros on the first tick (matches the dataset's i=0).  This keeps deploy
in-distribution with training (the dataset's prev-action was also a realized
pose delta, not a clamped command).

Defaults --checkpoint to the prev-action run; otherwise takes the SAME CLI as
run_policy (--go / --dry-run / --self-test, --goto-start, --task, --chain,
--force-limit, --max-ticks, --box, --no-home, --pause-between).

Run with the vla_venv python (case PRESENT, hand on the e-stop; do NOT run while
training shares the GPU):
  /home/dexmate/vla_venv/bin/python Research/contact_aware_vla/run_policy_prevaction.py \
      --go --goto-start LGES/recordings/case_pick/<take_dir> \
      --task case_pick --force-limit 15 --max-ticks 700

For the limit-cycle comparison, run the BASELINE with the stock executor instead:
  /home/dexmate/vla_venv/bin/python LGES/vla_training/run_policy.py --go \
      --checkpoint LGES/vla_training/outputs/smolvla_baseline_0617/checkpoints/last ...
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))
import run_policy  # noqa: E402  (no robot/heavy side effects at import)
from run_policy import _to_chw  # noqa: E402
from convert_to_lerobot import quat_mul, quat_conj, quat_to_rotvec  # noqa: E402

DEFAULT_CKPT = HERE / "outputs" / "smolvla_prevaction_0617" / "checkpoints" / "last"

# previous observed pose, to derive the realized previous-action delta.
_prev = {"pos": None, "quat": None}


def _reset_prev():
    _prev["pos"], _prev["quat"] = None, None


def _predict_prevaction(policy, pre, post, state, image, instruction, depth_image=None):
    """Drop-in for run_policy.predict that feeds the 22-dim (15 + prev action) obs.

    `state` is run_policy's 15-dim observation (ObsBuilder.state). We append the
    previous action and run the same select_action path."""
    pos, quat, suction = state[0:3], state[3:7], float(state[7])
    if _prev["pos"] is None:
        prev_action = np.zeros(7, dtype=np.float32)
    else:
        dpos = pos - _prev["pos"]
        drot = quat_to_rotvec(quat_mul(quat, quat_conj(_prev["quat"])))
        prev_action = np.concatenate([dpos, drot, [suction]]).astype(np.float32)
    state22 = np.concatenate([state, prev_action]).astype(np.float32)

    obs = {
        "observation.images.head": _to_chw(image),
        "observation.state": torch.from_numpy(state22).unsqueeze(0),
        "task": instruction,
    }
    if depth_image is not None:
        obs["observation.images.head_depth"] = _to_chw(depth_image)
    obs = pre(obs)
    with torch.inference_mode():
        action = policy.select_action(obs)
    _prev["pos"], _prev["quat"] = pos.copy(), quat.copy()
    return post(action).squeeze(0).cpu().numpy()


def main():
    argv = sys.argv[1:]
    if "--checkpoint" not in argv:
        if not DEFAULT_CKPT.exists():
            sys.exit(f"default checkpoint not found: {DEFAULT_CKPT}\n"
                     f"pass --checkpoint <prev-action run>/checkpoints/last")
        argv = ["--checkpoint", str(DEFAULT_CKPT)] + argv

    _reset_prev()
    run_policy.predict = _predict_prevaction  # patch only the obs/predict step
    sys.argv = ["run_policy.py"] + argv
    run_policy.main()


if __name__ == "__main__":
    main()
