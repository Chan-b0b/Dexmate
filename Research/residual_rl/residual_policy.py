#!/usr/bin/env python
"""Residual policy: a small Gaussian correction on top of the frozen base SmolVLA action.

Only the 6-d continuous EE pose delta (dpos 3 + drot 3) gets a residual; suction (the
policy's remaining action dim) passes through from the base policy unchanged -- it's
discrete and already reliable, the covariate-shift problem this is meant to fix is in
the continuous pose delta. Gaussian means this head has an exact, tractable log-prob for
AWR, unlike the base policy's one-step flow-matching sample (see
LGES/vla_training/smolvla_meanflow/README.md, "Notes for RL fine-tuning afterwards").

Conditions on `observation.state` only (15-d: pos, quat_wxyz, suction, sealed, wrench6) --
same "proprio" feature choice the ICM reward defaults to, so the actor's hot loop never
needs a second vision encode.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

POSE_DIM = 6  # dpos(3) + drot(3)


class ResidualPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int = 15,
        hidden_dim: int = 256,
        num_layers: int = 2,
        max_dpos_m: float = 0.003,
        max_drot_rad: float = 0.008,
        log_std_min: float = -5.0,
        log_std_max: float = 0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_dpos_m = max_dpos_m
        self.max_drot_rad = max_drot_rad
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.register_buffer(
            "max_delta",
            torch.tensor([max_dpos_m] * 3 + [max_drot_rad] * 3, dtype=torch.float32),
        )

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2 * POSE_DIM))  # [mean, log_std]
        self.net = nn.Sequential(*layers)
        # Zero-init the weights and the mean-bias: residual starts at exactly zero mean
        # (pure base policy) regardless of input. Bias the log_std half to log_std_min
        # (not 0) so it starts at its smallest allowed std, not the mid-range std that a
        # bare zero-init would give -- a freshly-initialized actor should be close to a
        # no-op on top of the frozen base until AWR actually moves it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[POSE_DIM:].fill_(log_std_min)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        out = self.net(obs)
        mean, log_std = out[..., :POSE_DIM], out[..., POSE_DIM:]
        mean = torch.tanh(mean) * self.max_delta
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp() * self.max_delta
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """obs: [..., obs_dim] -> raw residual pose delta [..., 6], already clamped to
        +-max_delta (the sampled/mean value can occasionally exceed it near the tanh
        saturation edge under a wide std)."""
        d = self.dist(obs)
        raw = d.mean if deterministic else d.sample()
        return raw.clamp(-self.max_delta, self.max_delta)

    def log_prob(self, obs: torch.Tensor, residual_action: torch.Tensor) -> torch.Tensor:
        """Sum of per-dim log-prob -> [...] (one scalar per sample), for the AWR loss."""
        return self.dist(obs).log_prob(residual_action).sum(dim=-1)

    def save(self, path: Path, extra: dict | None = None):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "max_dpos_m": self.max_dpos_m,
                    "max_drot_rad": self.max_drot_rad,
                    "log_std_min": self.log_std_min,
                    "log_std_max": self.log_std_max,
                },
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "ResidualPolicy":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        policy = cls(**ckpt["config"])
        policy.load_state_dict(ckpt["state_dict"])
        policy.loaded_extra = ckpt.get("extra", {})
        return policy


def save_atomic(policy: "ResidualPolicy", path: Path, extra: dict | None = None):
    """Write-then-rename so a concurrently-polling actor never reads a half-written
    checkpoint (the learner calls this; the actor only ever calls ResidualPolicy.load)."""
    tmp = Path(path).with_suffix(Path(path).suffix + ".tmp")
    policy.save(tmp, extra=extra)
    tmp.replace(path)
