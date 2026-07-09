# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""End-to-end check of a converted smolvla_meanflow checkpoint, exercising the exact code
path lerobot's tooling uses: plugin discovery -> config registry -> from_pretrained ->
pre/post processors -> one-step action generation.

Run:  HF_HUB_OFFLINE=1 python tests/test_checkpoint_load.py [checkpoint_dir]
"""

import sys
import time

import torch

# What `--policy.discover_packages_path=smolvla_meanflow` does inside lerobot-train:
from lerobot.configs.parser import load_plugin

load_plugin("smolvla_meanflow")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/home/maverick/checkpoints/smolvla_meanflow_base"


def main():
    print(f"Loading config from {CKPT} ...")
    cfg = PreTrainedConfig.from_pretrained(CKPT)
    assert cfg.type == "smolvla_meanflow", cfg.type
    cfg.pretrained_path = CKPT

    print("Loading policy through the factory ...")
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(CKPT, config=cfg)
    policy.eval()

    print("Loading processors saved next to the checkpoint ...")
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=CKPT,
        # lerobot-train/record override the device the same way at runtime.
        preprocessor_overrides={"device_processor": {"device": cfg.device or "cpu"}},
    )

    raw_batch = {
        "observation.state": torch.zeros(6),
        "observation.images.camera1": torch.rand(3, 256, 256),
        "observation.images.camera2": torch.rand(3, 256, 256),
        "observation.images.camera3": torch.rand(3, 256, 256),
        "task": "pick up the case",
    }
    batch = preprocessor(raw_batch)

    print(f"Generating an action chunk with num_steps={cfg.num_steps} ...")
    start = time.perf_counter()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    elapsed = time.perf_counter() - start
    assert chunk.shape == (1, cfg.chunk_size, 6), chunk.shape
    assert torch.isfinite(chunk).all()

    action = postprocessor(chunk[:, 0])
    print(f"OK: chunk {tuple(chunk.shape)} in {elapsed:.2f}s (CPU), first action after postprocess: "
          f"{action.squeeze().tolist()}")


if __name__ == "__main__":
    main()
