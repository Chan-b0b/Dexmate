#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Intrinsic Curiosity Module (forward-model ensemble) for reversed-curiosity RL.

The ICM is pretrained on demo data and then FROZEN: during RL fine-tuning its
prediction error measures distance from the demo manifold, and the (reversed)
reward is r = exp(-eta * error) in (0, 1].

Feature variants (selected at training time, recorded in the checkpoint):
  - proprio : normalized observation.state                       (15-d here)
  - vision  : frozen SmolVLA vision tower, mean-pooled tokens    (960-d/cam)
  - both    : concat of the two

The forward model predicts the *normalized feature delta* phi(s_t+1) - phi(s_t)
from [phi(s_t), a_t]. An ensemble of K independently initialized MLPs gives both
a mean prediction error and a disagreement (epistemic) signal — with small demo
datasets, disagreement is often better calibrated than single-model error.
"""

from __future__ import annotations

import torch
from torch import nn

VARIANTS = ("proprio", "vision", "both")


class ForwardModel(nn.Module):
    """MLP: [phi(s_t), a_t] -> predicted normalized delta phi."""

    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feat_dim + action_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feat_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat, action], dim=-1))


class ICMEnsemble(nn.Module):
    """Ensemble of forward models + normalization stats + reward computation.

    All inputs to the public API are RAW (unnormalized) features/actions; the
    module owns the normalization buffers so RL-time reward computation cannot
    silently diverge from how the ICM was trained.
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        variant: str = "proprio",
        ensemble_size: int = 5,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.members = nn.ModuleList(
            ForwardModel(feat_dim, action_dim, hidden_dim, num_layers) for _ in range(ensemble_size)
        )
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))
        self.register_buffer("act_mean", torch.zeros(action_dim))
        self.register_buffer("act_std", torch.ones(action_dim))
        # reward scale, set by calibration after training (see fit stats in train_icm.py)
        self.register_buffer("eta", torch.tensor(1.0))

    @torch.no_grad()
    def set_normalization(self, feat_mean, feat_std, act_mean, act_std):
        self.feat_mean.copy_(torch.as_tensor(feat_mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(feat_std, dtype=torch.float32).clamp_min(1e-6))
        self.act_mean.copy_(torch.as_tensor(act_mean, dtype=torch.float32))
        self.act_std.copy_(torch.as_tensor(act_std, dtype=torch.float32).clamp_min(1e-6))

    def normalize(self, feat: torch.Tensor, action: torch.Tensor):
        return (feat - self.feat_mean) / self.feat_std, (action - self.act_mean) / self.act_std

    def target_delta(self, feat: torch.Tensor, next_feat: torch.Tensor) -> torch.Tensor:
        """Normalized feature delta — the forward models' regression target."""
        return (next_feat - feat) / self.feat_std

    def predict(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Stacked member predictions of the normalized delta: [K, B, feat_dim]."""
        nfeat, nact = self.normalize(feat, action)
        return torch.stack([m(nfeat, nact) for m in self.members])

    def member_losses(self, feat, action, next_feat) -> torch.Tensor:
        """Per-member mean MSE (for training): [K]."""
        target = self.target_delta(feat, next_feat)
        preds = self.predict(feat, action)
        return ((preds - target.unsqueeze(0)) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def prediction_error(self, feat, action, next_feat) -> torch.Tensor:
        """Ensemble-mean squared error per sample: [B]. The OOD-ness signal."""
        target = self.target_delta(feat, next_feat)
        preds = self.predict(feat, action)
        return ((preds - target.unsqueeze(0)) ** 2).mean(dim=(0, 2))

    @torch.no_grad()
    def disagreement(self, feat, action) -> torch.Tensor:
        """Ensemble variance per sample: [B]. Epistemic signal, needs no next state."""
        preds = self.predict(feat, action)
        return preds.var(dim=0).mean(dim=-1)

    @torch.no_grad()
    def reward(self, feat, action, next_feat) -> torch.Tensor:
        """Reversed-curiosity reward r = exp(-eta * error) in (0, 1]: [B]."""
        return torch.exp(-self.eta * self.prediction_error(feat, action, next_feat))

    # -- persistence ---------------------------------------------------------

    def save(self, path, extra: dict | None = None):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "feat_dim": self.feat_dim,
                    "action_dim": self.action_dim,
                    "variant": self.variant,
                    "ensemble_size": len(self.members),
                    "hidden_dim": self.members[0].net[0].out_features,
                    "num_layers": sum(isinstance(m, nn.SiLU) for m in self.members[0].net),
                },
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu") -> "ICMEnsemble":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        icm = cls(**ckpt["config"])
        icm.load_state_dict(ckpt["state_dict"])
        icm.eval()
        icm.loaded_extra = ckpt.get("extra", {})
        return icm
