#!/usr/bin/env python3
"""Build a prev-action-augmented LeRobot dataset (contact_aware_vla experiment).

state = [original 15-dim] + [previous action (7-dim): prev dpos(3), prev drot(3),
prev suction(1)], where prev = action[t-1] (zeros at t=0).

Hypothesis (from the descend<->lift limit-cycle analysis): a mid-air observation
does not reveal the hidden grasp PHASE (pre-grasp descent vs post-grasp lift),
but the previous action does. Feeding it should let the policy commit to a
direction and break the mid-air chatter.

Reuses LGES/vla_training/convert_to_lerobot.load_take WITHOUT modifying it and
writes a NEW dataset under Research/, so the existing pipeline is untouched.
SmolVLA pads state to max_state_dim=32, so 15->22 needs no model change.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/contact_aware_vla/convert_prevaction.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))
from convert_to_lerobot import (  # noqa: E402
    load_take, colorize_depth, IMG_W, IMG_H, FPS,
    STATE_NAMES, ACTION_NAMES, IMAGE_FEATURE, SUCTION_TASKS)

PREV_NAMES = ["prev_dx", "prev_dy", "prev_dz", "prev_drx", "prev_dry", "prev_drz", "prev_suction"]
NEW_STATE_NAMES = STATE_NAMES + PREV_NAMES  # 15 + 7 = 22

MANIFEST = "processed_takes.json"


def load_manifest(root: Path) -> set:
    f = root / MANIFEST
    return set(json.loads(f.read_text())) if f.exists() else set()


def save_manifest(root: Path, processed: set):
    (root / MANIFEST).write_text(json.dumps(sorted(processed), indent=2))


def build_features(with_depth: bool) -> dict:
    feats = {
        "observation.images.head": dict(IMAGE_FEATURE),
        "observation.state": {"dtype": "float32", "shape": (len(NEW_STATE_NAMES),),
                              "names": NEW_STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    if with_depth:
        feats["observation.images.head_depth"] = dict(IMAGE_FEATURE)
    return feats


def write_episode(ds, instruction, rgb_paths, depth_paths, states, actions):
    for i in range(len(actions)):  # last frame dropped (no next-state action)
        img = cv2.cvtColor(cv2.resize(cv2.imread(str(rgb_paths[i])), (IMG_W, IMG_H)),
                           cv2.COLOR_BGR2RGB)
        prev = actions[i - 1] if i >= 1 else np.zeros(7, dtype=np.float32)
        state_aug = np.concatenate([states[i], prev]).astype(np.float32)  # 22-dim
        frame = {"observation.images.head": img, "observation.state": state_aug,
                 "action": actions[i], "task": instruction}
        if depth_paths is not None:
            depth_mm = cv2.imread(str(depth_paths[i]), cv2.IMREAD_UNCHANGED)
            frame["observation.images.head_depth"] = colorize_depth(depth_mm)
        ds.add_frame(frame)
    ds.save_episode()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=VLA_TRAIN.parent / "recordings")
    ap.add_argument("--out", type=Path, default=HERE / "datasets")
    ap.add_argument("--tasks", nargs="+", default=SUCTION_TASKS)
    ap.add_argument("--val-per-task", type=int, default=2)
    ap.add_argument("--name", default="lges_suction_prevaction")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--interventions", type=Path,
                    default=HERE.parent / "intervention" / "interventions",
                    help="dir with intervention takes (intervention_<task>/...); appended to TRAIN")
    ap.add_argument("--oversample", type=int, default=3,
                    help="write each intervention take this many times (DAgger up-weighting)")
    ap.add_argument("--incremental", action="store_true",
                    help="skip already-converted takes (train only); rebuild val always. "
                         "Rerun without this flag if you change --oversample.")
    args = ap.parse_args()
    with_depth = not args.no_depth

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    splits = {"train": [], "val": []}
    for task in args.tasks:
        takes = sorted(p for p in (args.recordings / task).iterdir() if p.is_dir())
        nv = args.val_per_task
        splits["train"] += takes[:-nv] if nv else takes
        splits["val"] += takes[-nv:] if nv else []

    # DAgger aggregation: append intervention takes to TRAIN only, oversampled K x.
    # (val stays demo-only for a clean, comparable eval across rounds.)
    n_iv = 0
    if args.interventions and args.interventions.exists():
        for task in args.tasks:
            idir = args.interventions / f"intervention_{task}"
            if not idir.exists():
                continue
            itakes = [p for p in sorted(idir.iterdir()) if p.is_dir()]
            splits["train"] += itakes * args.oversample      # duplicate K x -> K episodes each
            n_iv += len(itakes)
        if n_iv:
            print(f"[aggregate] +{n_iv} intervention takes x{args.oversample} "
                  f"= {n_iv * args.oversample} extra train episodes (failed/marked-n excluded by load_take)")

    for split, takes in splits.items():
        if not takes:
            continue
        name = args.name if split == "train" else f"{args.name}_val"
        root = args.out / name
        incremental = args.incremental and split == "train" and root.exists()

        if incremental:
            processed = load_manifest(root)
            ds = LeRobotDataset.resume(repo_id=f"local/{name}", root=root,
                                       image_writer_threads=12)
            print(f"[{split}] resuming {root} ({len(processed)} takes already converted)")
        else:
            if root.exists():
                shutil.rmtree(root)
            processed = set()
            ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=FPS,
                                       features=build_features(with_depth), root=root,
                                       robot_type="dexmate_vega_1p", use_videos=False,
                                       image_writer_threads=12)

        nf = 0
        for td in takes:
            take_key = f"{td.parent.name}/{td.name}"
            if take_key in processed:
                print(f"  [{split}] skip {take_key}")
                continue
            L = load_take(td, with_depth)
            if L is None:
                continue
            instr, rgb, dep, states, actions = L
            write_episode(ds, instr, rgb, dep, states, actions)
            processed.add(take_key)
            nf += len(actions)
            print(f"  [{split}] {td.parent.name}/{td.name}: {len(actions)} frames")

        save_manifest(root, processed)
        ds.finalize()
        print(f"{split}: {ds.num_episodes} eps, {nf} frames (state dim {len(NEW_STATE_NAMES)}) -> {root}")


if __name__ == "__main__":
    main()


#/home/dexmate/vla_venv/bin/python Research/contact_aware_vla/convert_prevaction.py --oversample 3
#RUN_NAME=smolvla_prevaction_dagger1 bash Research/contact_aware_vla/train_prevaction.sh --steps=20000
#/home/dexmate/vla_venv/bin/python Research/contact_aware_vla/convert_prevaction.py --incremental
