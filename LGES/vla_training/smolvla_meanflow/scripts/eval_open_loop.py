#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Open-loop offline eval for smolvla OR smolvla_meanflow checkpoints.

Mirrors vla_training/eval_offline.py's methodology (reset per episode, feed frames
through the checkpoint's own pre/post processors, select_action, MAE vs recorded
actions) but resolves the policy class from the checkpoint's config type, so the
same script evaluates stock SmolVLA and MeanFlow checkpoints. Also reports the
wall-clock time per action-chunk generation.

  python eval_open_loop.py --checkpoint <...>/checkpoints/last/pretrained_model \
      --dataset-root <lerobot dataset dir> --repo-id <repo id> [--max-episodes 8]
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import smolvla_meanflow  # noqa: F401  (registers the smolvla_meanflow policy type)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="pretrained_model dir")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--max-episodes", type=int, default=8, help="evenly-spaced episode subset")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="override inference NFEs (meanflow: >1 trades latency for accuracy)")
    args = ap.parse_args()

    model_dir = args.checkpoint
    cfg = PreTrainedConfig.from_pretrained(str(model_dir))
    policy_cls = get_policy_class(cfg.type)
    print(f"checkpoint: {model_dir}  (type={cfg.type}, class={policy_cls.__name__}, "
          f"num_steps={getattr(cfg, 'num_steps', '?')})")

    policy = policy_cls.from_pretrained(str(model_dir))
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps
        cfg.num_steps = args.num_steps
    policy.eval()
    device = policy.config.device
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    bounds = [(ep["dataset_from_index"], ep["dataset_to_index"]) for ep in ds.meta.episodes]
    if len(bounds) > args.max_episodes:
        idx = np.linspace(0, len(bounds) - 1, args.max_episodes).astype(int)
        bounds = [bounds[i] for i in idx]
    print(f"eval: {len(bounds)} episodes (of {ds.num_episodes}), dataset {args.repo_id}")

    per_task = defaultdict(list)
    chunk_times = []

    for lo, hi in bounds:
        policy.reset()
        task = ds[lo]["task"]
        errs = []
        for i in range(lo, hi):
            frame = ds[i]
            obs = {
                "observation.images.head": frame["observation.images.head"].unsqueeze(0),
                "observation.state": frame["observation.state"].unsqueeze(0),
                "task": task,
            }
            if "observation.images.head_depth" in frame:
                obs["observation.images.head_depth"] = frame["observation.images.head_depth"].unsqueeze(0)
            obs = pre(obs)
            queue_empty = len(policy._queues["action"]) == 0
            t0 = time.perf_counter()
            with torch.inference_mode():
                action = policy.select_action(obs)
            if queue_empty:  # this call generated a fresh chunk
                chunk_times.append(time.perf_counter() - t0)
            action = post(action).squeeze(0).cpu().numpy()
            errs.append(np.abs(action - frame["action"].numpy()))
        per_task[task].append(np.array(errs))

    all_errs = np.concatenate([np.concatenate(eps) for eps in per_task.values()])
    adim = all_errs.shape[1]
    # rel (7): dx dy dz | drx dry drz | suction    abs (8): x y z | qw qx qy qz | suction
    rot_sl = slice(3, adim - 1)
    print("-" * 64)
    print(f"episodes={len(bounds)}  frames={len(all_errs)}  action_dim={adim}")
    print(f"pos MAE      : {all_errs[:, :3].mean() * 1000:8.3f} mm/step")
    print(f"rot MAE      : {all_errs[:, rot_sl].mean() * 1000:8.3f} x1e-3")
    print(f"suction acc  : {(all_errs[:, -1] < 0.5).mean() * 100:7.1f} %")
    print(f"per-dim MAE  : {np.array2string(all_errs.mean(0), precision=5)}")
    if chunk_times:
        ct = np.array(chunk_times[1:] or chunk_times)  # drop warmup
        print(f"chunk gen    : {ct.mean() * 1000:7.1f} ms avg over {len(ct)} chunks "
              f"({getattr(cfg, 'num_steps', '?')} denoise step(s))")


if __name__ == "__main__":
    main()
