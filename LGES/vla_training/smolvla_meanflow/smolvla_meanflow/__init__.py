# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""SmolVLA + MeanFlow: one-step action generation for LeRobot, as a third-party plugin.

Importing this package registers the ``smolvla_meanflow`` policy type with LeRobot's
config registry. Nothing in the installed ``lerobot`` package is modified.

Usage with lerobot-train:
    lerobot-train \
      --policy.type=smolvla_meanflow \
      --policy.discover_packages_path=smolvla_meanflow \
      --dataset.repo_id=... \
      ...

References:
- MeanFlow: Geng et al., "Mean Flows for One-step Generative Modeling", arXiv:2505.13447
- MF-VLA: "Mean-Flow based One-Step Vision-Language-Action", arXiv:2603.01469
"""

from smolvla_meanflow.configuration_smolvla_meanflow import SmolVLAMeanFlowConfig
from smolvla_meanflow.modeling_smolvla_meanflow import SmolVLAMeanFlowPolicy, VLAMeanFlow
from smolvla_meanflow.processor_smolvla_meanflow import make_smolvla_meanflow_pre_post_processors

__all__ = [
    "SmolVLAMeanFlowConfig",
    "SmolVLAMeanFlowPolicy",
    "VLAMeanFlow",
    "make_smolvla_meanflow_pre_post_processors",
]
