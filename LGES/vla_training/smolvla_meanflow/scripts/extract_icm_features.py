#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Precompute ICM features for a LeRobot dataset into a single .npz cache.

Stores per-frame: proprio state, action, episode_index, and (unless --no-vision)
frozen SmolVLA vision features (embed_image tokens mean-pooled -> 960-d/camera,
fp16). train_icm.py then trains any variant (proprio/vision/both) from this one
cache without touching the policy again.

Vision preprocessing mirrors SmolVLA's prepare_images exactly: resize_with_pad
to config.resize_imgs_with_padding, then [0,1] -> [-1,1].

  python extract_icm_features.py \
      --checkpoint <...>/checkpoints/last/pretrained_model \
      --dataset-root <lerobot dataset dir> --repo-id <repo id> \
      --out icm_features_0708.npz [--batch-size 64] [--max-frames N]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import smolvla_meanflow  # noqa: F401  (registers the smolvla_meanflow policy type)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, help="pretrained_model dir (vision tower source; required unless --no-vision)")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-vision", action="store_true", help="proprio/action/episode arrays only (CPU-fast)")
    ap.add_argument("--max-frames", type=int, default=None, help="smoke-test cap")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    n = min(len(ds), args.max_frames) if args.max_frames else len(ds)
    image_keys = [k for k in ds.meta.features if k.startswith("observation.images.") and "depth" not in k]
    print(f"dataset: {args.repo_id}  frames={n}/{len(ds)}  cameras={image_keys}")

    vision_model, resize_hw, device = None, None, "cpu"
    if not args.no_vision:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --no-vision is set")
        cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
        policy = get_policy_class(cfg.type).from_pretrained(str(args.checkpoint))
        policy.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vision_model = policy.model.vlm_with_expert
        vision_model.to(device)
        resize_hw = cfg.resize_imgs_with_padding
        print(f"vision tower from {args.checkpoint} (type={cfg.type}), device={device}, resize={resize_hw}")

    states, actions, episodes, vision_feats = [], [], [], []
    batch_imgs: dict[str, list] = {k: [] for k in image_keys}

    def flush_vision():
        if vision_model is None or not batch_imgs[image_keys[0]]:
            return
        per_cam = []
        for k in image_keys:
            imgs = torch.stack(batch_imgs[k]).to(device)  # [B,3,H,W] in [0,1]
            imgs = resize_with_pad(imgs, *resize_hw, pad_value=0)
            imgs = imgs * 2.0 - 1.0
            with torch.inference_mode():
                tokens = vision_model.embed_image(imgs)  # [B, T, D]
            per_cam.append(tokens.mean(dim=1))  # mean-pool tokens -> [B, D]
            batch_imgs[k].clear()
        vision_feats.append(torch.cat(per_cam, dim=-1).to(torch.float16).cpu().numpy())

    for i in range(n):
        frame = ds[i]
        states.append(frame["observation.state"].numpy())
        actions.append(frame["action"].numpy())
        episodes.append(int(frame["episode_index"]))
        if vision_model is not None:
            for k in image_keys:
                batch_imgs[k].append(frame[k])
            if len(batch_imgs[image_keys[0]]) >= args.batch_size:
                flush_vision()
                print(f"  {i + 1}/{n} frames", flush=True)
    flush_vision()

    out = {
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "episode_index": np.asarray(episodes, dtype=np.int64),
    }
    if vision_feats:
        out["vision"] = np.concatenate(vision_feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    shapes = {k: v.shape for k, v in out.items()}
    print(f"wrote {args.out}  {shapes}")


if __name__ == "__main__":
    main()
