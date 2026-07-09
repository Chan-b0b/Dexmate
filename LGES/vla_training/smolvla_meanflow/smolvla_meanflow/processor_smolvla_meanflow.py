# Copyright 2026. Licensed under the Apache License, Version 2.0.
from typing import Any

import torch

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from smolvla_meanflow.configuration_smolvla_meanflow import SmolVLAMeanFlowConfig


def make_smolvla_meanflow_pre_post_processors(
    config: SmolVLAMeanFlowConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """MeanFlow only changes the training objective and sampler; pre/post-processing
    (rename, tokenize, normalize, device) is identical to stock SmolVLA."""
    return make_smolvla_pre_post_processors(config=config, dataset_stats=dataset_stats)
