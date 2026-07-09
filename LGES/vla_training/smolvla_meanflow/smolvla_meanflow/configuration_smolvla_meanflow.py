# Copyright 2026. Licensed under the Apache License, Version 2.0.
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_meanflow")
@dataclass
class SmolVLAMeanFlowConfig(SmolVLAConfig):
    """SmolVLA config with MeanFlow (average-velocity) training and few-step inference.

    The model learns u(x_t, r, t), the average velocity over the interval [r, t]
    (SmolVLA time convention: t=1 is noise, t=0 is data), so inference needs only
    ``num_steps`` forward passes (1 by default) instead of SmolVLA's 10 Euler steps.
    """

    # MeanFlow inference: number of function evaluations. 1 = one-step generation.
    num_steps: int = 1

    # Fraction of training samples with r < t (the MeanFlow bootstrap samples). The rest
    # are trained with r == t, which reduces to plain flow matching. Papers report best
    # results with a minority of r != t samples (~20-25%).
    meanflow_time_diff_ratio: float = 0.25

    # Adaptive loss weighting w = 1 / (||delta||^2 + c)^(1 - gamma) from the MeanFlow paper.
    # gamma = 1.0 disables the weighting (plain L2).
    meanflow_adaptive_gamma: float = 0.5
    meanflow_adaptive_c: float = 1e-3

    # Zero-init the interval-conditioning projection so that at initialization
    # u(x, t, t) exactly equals the instantaneous velocity of a warm-start SmolVLA
    # checkpoint. Keep True when initializing from pretrained SmolVLA weights.
    zero_init_interval_proj: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.rtc_config is not None:
            raise NotImplementedError(
                "Real-Time Chunking is not supported with MeanFlow (it modulates instantaneous "
                "velocities over many denoising steps, which one-step inference removes)."
            )
        if not self.use_cache:
            raise ValueError(
                "smolvla_meanflow requires use_cache=True: both the training JVP and sampling "
                "reuse the prefix KV cache."
            )
        if not 0.0 <= self.meanflow_time_diff_ratio <= 1.0:
            raise ValueError(
                f"meanflow_time_diff_ratio must be in [0, 1], got {self.meanflow_time_diff_ratio}"
            )
        if self.compile_model:
            raise NotImplementedError(
                "compile_model is not supported: torch.func.jvp used by the MeanFlow loss "
                "does not compose with torch.compile of the full forward."
            )
