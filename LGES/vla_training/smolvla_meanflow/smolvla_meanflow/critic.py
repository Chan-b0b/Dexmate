#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""State-value critic for AWR fine-tuning and regret-based curriculum.

V(s) is trained by regression on normalized discounted returns of the
reversed-curiosity reward r = exp(-eta * ICM error) in (0, 1]:

    v_t = (1 - gamma) * r_t + gamma * v_{t+1},   v_last = r_last  (absorbing tail)

The (1 - gamma) normalization keeps targets in (0, 1] — V reads as "discounted
average on-manifold-ness of the near future" — and the absorbing tail removes
the end-of-episode artifact (without it, states late in an episode look low-value
just because few reward steps remain).

A small ensemble gives: mean -> V for AWR advantages; per-member spread and
positive value loss -> regret/uncertainty signals for the PLR/ACCEL curriculum.
"""

from __future__ import annotations

import torch
from torch import nn


class ValueMLP(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feat_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)


class CriticEnsemble(nn.Module):
    """Ensemble value function over the same raw features the ICM uses."""

    def __init__(
        self,
        feat_dim: int,
        variant: str = "proprio",
        ensemble_size: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.variant = variant
        self.feat_dim = feat_dim
        self.members = nn.ModuleList(
            ValueMLP(feat_dim, hidden_dim, num_layers) for _ in range(ensemble_size)
        )
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    @torch.no_grad()
    def set_normalization(self, feat_mean, feat_std):
        self.feat_mean.copy_(torch.as_tensor(feat_mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(feat_std, dtype=torch.float32).clamp_min(1e-6))

    def predict(self, feat: torch.Tensor) -> torch.Tensor:
        """Stacked member values: [K, B]."""
        nfeat = (feat - self.feat_mean) / self.feat_std
        return torch.stack([m(nfeat) for m in self.members])

    @torch.no_grad()
    def value(self, feat: torch.Tensor) -> torch.Tensor:
        """Ensemble-mean V(s): [B]."""
        return self.predict(feat).mean(dim=0)

    @torch.no_grad()
    def uncertainty(self, feat: torch.Tensor) -> torch.Tensor:
        """Ensemble std of V(s): [B]. Epistemic signal for curriculum gating."""
        return self.predict(feat).std(dim=0)

    @staticmethod
    def normalized_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
        """Absorbing-tail normalized returns for ONE episode's rewards [T] -> [T]."""
        v = torch.empty_like(rewards)
        v[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            v[t] = (1.0 - gamma) * rewards[t] + gamma * v[t + 1]
        return v

    # -- persistence ---------------------------------------------------------

    def save(self, path, extra: dict | None = None):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "feat_dim": self.feat_dim,
                    "variant": self.variant,
                    "ensemble_size": len(self.members),
                    "hidden_dim": self.members[0].net[0].out_features,
                    "num_layers": sum(isinstance(m, nn.SiLU) for m in self.members[0].net),
                    "gamma": float(self.gamma),
                },
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu") -> "CriticEnsemble":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        critic = cls(**ckpt["config"])
        critic.load_state_dict(ckpt["state_dict"])
        critic.eval()
        critic.loaded_extra = ckpt.get("extra", {})
        return critic
