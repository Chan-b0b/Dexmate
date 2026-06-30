#!/usr/bin/env python3
"""Latent dynamics model for the LGES task (TD-MPC2-flavored).

  enc:  state[15]      -> z[latent]      (SimNorm-bounded)
  dyn:  (z, action[7]) -> z'[latent]     (SimNorm-bounded)
  dec:  z              -> state_hat[15]   (normalized space)

Trained purely offline on demos with three losses (no reward/value yet — that
is the later planner stage):
  * decode loss   : decode(z_t) matches the true (normalized) state
  * consistency   : z_t matches enc(true_state_t) detached  (keeps latent
                    rollouts on the encoded manifold — critical for planning)
  * reconstruction: decode(enc(s_0)) matches s_0  (grounds the autoencoder)

The decoder predicts normalized pos(3)+quat(4)+wrench(6) as regression and
suction(7)/sealed(8) as logits — see data.py for the index map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import STATE_DIM, ACTION_DIM, POS, QUAT, SUCTION, SEALED, WRENCH


class SimNorm(nn.Module):
    """TD-MPC2 simplicial normalization: softmax over groups of size V.

    Keeps the latent bounded and sparse, which stops multi-step rollouts from
    drifting/exploding."""

    def __init__(self, V=8):
        super().__init__()
        self.V = V

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], shp[-1] // self.V, self.V)
        x = F.softmax(x, dim=-1)
        return x.reshape(*shp)


def _mlp(sizes, act=nn.Mish):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class LatentDynamics(nn.Module):
    def __init__(self, latent=256, hidden=512, simnorm_v=8):
        super().__init__()
        assert latent % simnorm_v == 0, "latent dim must be divisible by simnorm V"
        self.latent = latent
        self.enc = nn.Sequential(_mlp([STATE_DIM, hidden, latent]), SimNorm(simnorm_v))
        self.dyn = nn.Sequential(_mlp([latent + ACTION_DIM, hidden, hidden, latent]),
                                 SimNorm(simnorm_v))
        self.dec = _mlp([latent, hidden, STATE_DIM])

    def encode(self, s):
        return self.enc(s)

    def step(self, z, a):
        return self.dyn(torch.cat([z, a], dim=-1))

    def decode(self, z):
        return self.dec(z)


def decode_losses(pred, target, seal_weight=2.0):
    """Per-group prediction losses on normalized state. Returns a dict; the
    `total` key weights the seal event up since it is the task-critical signal."""
    l_pos = F.mse_loss(pred[..., POS], target[..., POS])
    qp = F.normalize(pred[..., QUAT], dim=-1)
    qt = target[..., QUAT]
    l_quat = (1.0 - (qp * qt).sum(-1) ** 2).mean()          # double-cover geodesic
    l_suction = F.binary_cross_entropy_with_logits(pred[..., SUCTION], target[..., SUCTION])
    l_sealed = F.binary_cross_entropy_with_logits(pred[..., SEALED], target[..., SEALED])
    l_wrench = F.mse_loss(pred[..., WRENCH], target[..., WRENCH])
    total = l_pos + l_quat + 0.5 * l_suction + seal_weight * l_sealed + l_wrench
    return {"pos": l_pos, "quat": l_quat, "suction": l_suction,
            "sealed": l_sealed, "wrench": l_wrench, "total": total}


def rollout_loss(model, states, actions, rho=0.9, w_consistency=10.0,
                 w_recon=1.0, seal_weight=2.0):
    """Multi-step world-model loss over a batch of sub-trajectories.

    states  : [B, H+1, 15] normalized
    actions : [B, H, 7]    normalized
    """
    H = actions.shape[1]
    z = model.encode(states[:, 0])

    # autoencoder grounding at t=0
    recon = decode_losses(model.decode(z), states[:, 0], seal_weight)
    agg = {k: w_recon * v for k, v in recon.items()}
    loss = w_recon * recon["total"]

    norm = w_recon
    for t in range(H):
        z = model.step(z, actions[:, t])
        tgt = states[:, t + 1]
        dl = decode_losses(model.decode(z), tgt, seal_weight)
        with torch.no_grad():
            z_tgt = model.encode(tgt)
        cons = F.mse_loss(z, z_tgt)
        w = rho ** t
        loss = loss + w * (dl["total"] + w_consistency * cons)
        norm += w
        for k, v in dl.items():
            agg[k] = agg.get(k, 0.0) + w * v
        agg["consistency"] = agg.get("consistency", 0.0) + w * cons

    loss = loss / norm
    agg = {k: (v / norm).item() for k, v in agg.items()}
    return loss, agg
