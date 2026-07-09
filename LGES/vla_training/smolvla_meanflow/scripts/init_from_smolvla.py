#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Warm-start a SmolVLAMeanFlowPolicy from a pretrained SmolVLA checkpoint.

Copies the config and all weights from a stock SmolVLA checkpoint (hub repo id or local
path) into a fresh smolvla_meanflow checkpoint directory. The only new parameters are the
zero-initialized interval projection, so right after conversion the model's u(x, t, t)
is bit-identical to the source model's instantaneous velocity v(x, t).

Also copies the pre/post-processor files (normalization stats, tokenizer settings) next to
the converted weights so `lerobot-train --policy.path=<out_dir>` works out of the box.

Example:
    python scripts/init_from_smolvla.py \
        --src lerobot/smolvla_base \
        --out ~/checkpoints/smolvla_meanflow_base \
        --num-steps 1
"""

import argparse
import dataclasses
import shutil
from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smolvla_meanflow import SmolVLAMeanFlowConfig, SmolVLAMeanFlowPolicy

PROCESSOR_FILE_PATTERNS = ("policy_preprocessor*", "policy_postprocessor*")


def resolve_source_dir(src: str) -> Path:
    """Return the local directory holding the source checkpoint files."""
    src_path = Path(src).expanduser()
    if src_path.is_dir():
        return src_path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(src))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="lerobot/smolvla_base", help="Hub repo id or local checkpoint dir")
    parser.add_argument("--out", required=True, help="Output directory for the smolvla_meanflow checkpoint")
    parser.add_argument("--num-steps", type=int, default=1, help="Inference NFE (1 = one-step)")
    parser.add_argument(
        "--meanflow-time-diff-ratio", type=float, default=0.25, help="Fraction of samples with r < t"
    )
    args = parser.parse_args()

    print(f"Loading source policy from {args.src} ...")
    src_policy = SmolVLAPolicy.from_pretrained(args.src)

    # Copy every SmolVLA config field into the MeanFlow config.
    smolvla_field_names = {f.name for f in dataclasses.fields(SmolVLAConfig) if f.init}
    cfg_kwargs = {name: getattr(src_policy.config, name) for name in smolvla_field_names}
    cfg_kwargs.update(
        num_steps=args.num_steps,
        meanflow_time_diff_ratio=args.meanflow_time_diff_ratio,
        zero_init_interval_proj=True,
        # Loading the VLM from the hub is pointless here since every weight is overwritten
        # by the source checkpoint right after construction.
        load_vlm_weights=False,
    )
    cfg = SmolVLAMeanFlowConfig(**cfg_kwargs)

    print("Building MeanFlow policy and transferring weights ...")
    dst_policy = SmolVLAMeanFlowPolicy(cfg)
    missing, unexpected = dst_policy.load_state_dict(src_policy.state_dict(), strict=False)

    if unexpected:
        raise RuntimeError(f"Unexpected keys when transferring weights: {unexpected}")
    not_interval = [k for k in missing if "action_interval_proj" not in k]
    if not_interval:
        raise RuntimeError(f"Missing keys other than the new interval projection: {not_interval}")
    print(f"Weights transferred. New zero-initialized params: {missing}")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_policy.save_pretrained(out_dir)

    src_dir = resolve_source_dir(args.src)
    copied = []
    for pattern in PROCESSOR_FILE_PATTERNS:
        for f in src_dir.glob(pattern):
            shutil.copy2(f, out_dir / f.name)
            copied.append(f.name)
    print(f"Copied processor files: {copied}")
    print(f"Done. Train with:\n"
          f"  lerobot-train \\\n"
          f"    --policy.path={out_dir} \\\n"
          f"    --policy.discover_packages_path=smolvla_meanflow \\\n"
          f"    --dataset.repo_id=<your dataset> ...")


if __name__ == "__main__":
    main()
