#!/usr/bin/env python3
"""Convert case_battery_demo recorded takes into a LeRobotDataset.

Scope (decided 2026-06-12): suction-only left-arm sub-tasks. Actions are
next-state EE deltas (the /tmp trace was lost to a reboot; observed deltas
are the standard substitute). Force enters the state as the RAW 6-axis
wrench — the tared value depends on script-side tare events the policy
cannot reproduce at deployment.

Per frame t (last frame of each take dropped — no next state):
  observation.images.head       : head RGB resized to 512x320
  observation.images.head_depth : head depth, turbo-colorized to 3ch (camera2);
                                  SigLIP needs RGB, so depth is rendered, not
                                  raw-metric. Disable with --no-depth.
  observation.state (15)  : ee pos(3) + ee quat wxyz(4, sign-continuous)
                            + suction(1) + vacuum_sealed(1, physical DI0 seal)
                            + raw wrench fx..tz(6)
  action (7)              : delta pos(3, base frame) + delta rot(3, rotvec
                            of R_{t+1} R_t^T, base frame) + suction_cmd[t+1]
  task                    : per-take instruction from meta.json

barcode_confirmed is logged by the recorder but intentionally NOT used here.

Frames are stored as images, not video: lerobot's pyav decode path needs
torchvision.io.VideoReader, removed in our torchvision 0.26, and torchcodec
has no aarch64 build. Image mode sidesteps decoding entirely.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python LGES/vla_training/convert_to_lerobot.py \
      [--recordings DIR] [--out DIR] [--val-per-task 2]
"""
#/home/dexmate/vla_venv/bin/python LGES/vla_training/convert_to_lerobot.py --recordings LGES/recordings/ --out LGES/vla_training/datasets/ --val-per-task 2


import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

VLA_DIR = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = VLA_DIR.parent / "recordings"
DEFAULT_OUT = VLA_DIR / "datasets"

SUCTION_TASKS = [
    "case_pick",
    "case_place",
    "battery_1_pick",
    "battery_1_place",
    "battery_2_pick",
    "battery_2_place",
]

IMG_W, IMG_H = 512, 320
FPS = 15
# Depth colorize span (mm), matching the dashboard publisher's _colorize_depth.
# Recorded head depth spans ~300-1000mm (workspace ~750mm); invalid (0) -> black.
DEPTH_NEAR_MM, DEPTH_FAR_MM = 300.0, 1000.0

STATE_NAMES = [
    "ee_x", "ee_y", "ee_z",
    "ee_qw", "ee_qx", "ee_qy", "ee_qz",
    "suction",
    "vacuum_sealed",
    "fx", "fy", "fz", "tx", "ty", "tz",
]
ACTION_NAMES_DELTA = ["dx", "dy", "dz", "drx", "dry", "drz", "suction"]
ACTION_NAMES_ABS = ["x", "y", "z", "qw", "qx", "qy", "qz", "suction"]

# Canonical quaternion sign anchor. The straight-down grasp (roll=pi) has qw~0,
# so a qw-sign rule is a per-episode coin flip: the same pose lands as qx~+1 in
# some episodes and qx~-1 in others (the training cost behind the "roll jumps
# between -178 and 178" symptom). Flip when dot(q, Q_REF) = qx < 0 instead —
# qx = cos(yaw/2) >= ~0.57 over the demo's yaw range, far from any knife edge.
# run_policy.ObsBuilder applies the SAME rule at deployment.
Q_REF = np.array([0.0, 1.0, 0.0, 0.0])

IMAGE_FEATURE = {"dtype": "image", "shape": (IMG_H, IMG_W, 3),
                 "names": ["height", "width", "channels"]}


def build_features(with_depth: bool, action_names) -> dict:
    feats = {
        "observation.images.head": dict(IMAGE_FEATURE),
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),),
                              "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(action_names),),
                   "names": list(action_names)},
    }
    if with_depth:
        feats["observation.images.head_depth"] = dict(IMAGE_FEATURE)
    return feats


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 mm depth -> turbo-colorized RGB (IMG_W x IMG_H). Invalid (0) ->
    black. Same mapping as dashboard publisher._colorize_depth so a deployed
    policy can reproduce it from live head depth."""
    d = depth_mm.astype(np.float32)
    valid = d > 0.0
    norm = np.clip((d - DEPTH_NEAR_MM) / (DEPTH_FAR_MM - DEPTH_NEAR_MM), 0.0, 1.0)
    norm[~valid] = 0.0
    u8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)  # BGR
    color[~valid] = (0, 0, 0)
    color = cv2.resize(color, (IMG_W, IMG_H))
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


# ── quaternion helpers (wxyz) ────────────────────────────────────────


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    if q[0] < 0:  # shortest path
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(s, q[0])
    return v / s * angle


# ── take loading ─────────────────────────────────────────────────────


def load_take(take_dir: Path, with_depth: bool, action_space: str = "delta"):
    """Return (instruction, rgb_paths, depth_paths|None, states[N,15],
    actions[N-1,7 delta | N-1,8 abs]) or None.

    action_space 'delta': next-state EE deltas (pos delta + rotvec + suction).
    action_space 'abs'  : next-state ABSOLUTE EE pose (pos + canonical quat
    wxyz + suction) — orientation stays a quaternion because the grasp is a
    ~180 deg rotation, exactly where an absolute rotvec is ill-conditioned."""
    meta = json.loads((take_dir / "meta.json").read_text())
    if meta.get("success") is not True:
        return None

    frames = [json.loads(line) for line in (take_dir / "states.jsonl").open()]
    rgb_paths = sorted((take_dir / "head_rgb").glob("*.jpg"))
    depth_paths = sorted((take_dir / "head_depth").glob("*.png")) if with_depth else None
    # A wrench read can race the take start (recorder logs wrench=null) — drop
    # such leading frames with their images; a null mid-take is still fatal.
    k = 0
    while k < len(frames) and frames[k].get("wrench") is None:
        k += 1
    if k:
        print(f"  {take_dir.name}: dropped {k} leading null-wrench frame(s)")
        frames, rgb_paths = frames[k:], rgb_paths[k:]
        if with_depth:
            depth_paths = depth_paths[k:]
    n = min(len(frames), len(rgb_paths))
    if with_depth:
        n = min(n, len(depth_paths))
    if n < 2:
        print(f"  SKIP {take_dir.name}: only {n} frames")
        return None
    frames, rgb_paths = frames[:n], rgb_paths[:n]
    if with_depth:
        depth_paths = depth_paths[:n]

    pos = np.array([f["ee"]["pos"] for f in frames], dtype=np.float64)
    quat = np.array([f["ee"]["quat_wxyz"] for f in frames], dtype=np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # sign continuity so the state input doesn't jump between q and -q
    for i in range(1, n):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    # canonical sign for the WHOLE episode (see Q_REF note above)
    if np.dot(quat[0], Q_REF) < 0:
        quat = -quat

    suction = np.array([1.0 if f["suction_cmd"] else 0.0 for f in frames])
    sealed = np.array([1.0 if f.get("vacuum_sealed") else 0.0 for f in frames])
    wrench = np.array(
        [[f["wrench"][k] for k in ("fx", "fy", "fz", "tx", "ty", "tz")] for f in frames],
        dtype=np.float64,
    )
    states = np.concatenate(
        [pos, quat, suction[:, None], sealed[:, None], wrench], axis=1
    ).astype(np.float32)

    if action_space == "abs":
        actions = np.concatenate([pos[1:], quat[1:], suction[1:, None]], axis=1).astype(np.float32)
    else:
        dpos = pos[1:] - pos[:-1]
        drot = np.stack(
            [quat_to_rotvec(quat_mul(quat[i + 1], quat_conj(quat[i]))) for i in range(n - 1)]
        )
        actions = np.concatenate([dpos, drot, suction[1:, None]], axis=1).astype(np.float32)

    return meta["instruction"], rgb_paths, depth_paths, states, actions


def write_episode(dataset, instruction, rgb_paths, depth_paths, states, actions):
    for i in range(len(actions)):  # last frame dropped (no next state)
        img = cv2.imread(str(rgb_paths[i]))
        img = cv2.cvtColor(cv2.resize(img, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
        frame = {
            "observation.images.head": img,
            "observation.state": states[i],
            "action": actions[i],
            "task": instruction,
        }
        if depth_paths is not None:
            depth_mm = cv2.imread(str(depth_paths[i]), cv2.IMREAD_UNCHANGED)
            frame["observation.images.head_depth"] = colorize_depth(depth_mm)
        dataset.add_frame(frame)
    dataset.save_episode()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tasks", nargs="+", default=SUCTION_TASKS)
    ap.add_argument("--val-per-task", type=int, default=2,
                    help="hold out the N most recent takes per task into <name>_val")
    ap.add_argument("--val-takes", nargs="+", default=None,
                    help="exact take directory names to force into val, overriding "
                         "--val-per-task's automatic last-N selection; the rest go to train")
    ap.add_argument("--name", default="lges_suction")
    ap.add_argument("--no-depth", action="store_true",
                    help="omit the colorized depth camera (depth is on by default)")
    ap.add_argument("--action-space", choices=("delta", "abs"), default="delta",
                    help="delta: next-state EE deltas (7d); abs: next-state "
                         "absolute EE pose pos+quat_wxyz+suction (8d)")
    args = ap.parse_args()
    with_depth = not args.no_depth
    action_names = ACTION_NAMES_ABS if args.action_space == "abs" else ACTION_NAMES_DELTA

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    splits = {"train": [], "val": []}
    for task in args.tasks:
        takes = sorted(p for p in (args.recordings / task).iterdir() if p.is_dir())
        if args.val_takes:
            val_names = set(args.val_takes)
            splits["val"] += [t for t in takes if t.name in val_names]
            splits["train"] += [t for t in takes if t.name not in val_names]
        else:
            n_val = args.val_per_task
            splits["train"] += takes[:-n_val] if n_val else takes
            splits["val"] += takes[-n_val:] if n_val else []

    for split, takes in splits.items():
        if not takes:
            continue
        name = args.name if split == "train" else f"{args.name}_val"
        root = args.out / name
        if root.exists():
            shutil.rmtree(root)
        dataset = LeRobotDataset.create(
            repo_id=f"local/{name}",
            fps=FPS,
            features=build_features(with_depth, action_names),
            root=root,
            robot_type="dexmate_vega_1p",
            use_videos=False,
        )
        n_frames = 0
        for take_dir in takes:
            loaded = load_take(take_dir, with_depth, args.action_space)
            if loaded is None:
                continue
            instruction, rgb_paths, depth_paths, states, actions = loaded
            write_episode(dataset, instruction, rgb_paths, depth_paths, states, actions)
            n_frames += len(actions)
            print(f"  [{split}] {take_dir.parent.name}/{take_dir.name}: {len(actions)} frames")
        dataset.finalize()
        print(f"{split}: {dataset.num_episodes} episodes, {n_frames} frames -> {root}")


if __name__ == "__main__":
    main()
