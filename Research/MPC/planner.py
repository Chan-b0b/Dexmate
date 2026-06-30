#!/usr/bin/env python3
"""MPPI planner over the latent dynamics model for LGES deploy-time MPC.

Pose is integrated ANALYTICALLY from the sampled action deltas (exact, and the
latent decode of position is imperfect); the latent model supplies the genuinely
learned signals the reward needs — seal probability and contact force. Same
(state[15] -> action[7]) interface as run_policy.py, so it can drop into that
loop. Receding horizon: plan() returns the first action and warm-starts the next
call.
"""

import numpy as np
import torch

from data import POS, SEALED, WRENCH, ACTION_DIM

# Per-step action bounds (match run_policy.py safety clamps); suction in [0,1].
A_LO = np.array([-0.01, -0.03, -0.03, -0.025, -0.025, -0.025, 0.0], np.float32)
A_HI = np.array([0.01, 0.03, 0.03, 0.025, 0.025, 0.025, 1.0], np.float32)
# Sampling std per action dim (~ the recorded per-step deltas, suction explores).
A_STD = np.array([0.004, 0.006, 0.006, 0.006, 0.007, 0.008, 0.3], np.float32)
_FZ = WRENCH.start + 2  # absolute index of fz in the state vector


class MPPIPlanner:
    def __init__(self, model, norm, reward, device, horizon=20, n_samples=512,
                 n_iters=4, temperature=0.5, gamma=0.95):
        self.model, self.reward, self.device = model, reward.to(device), device
        self.H, self.N, self.iters = horizon, n_samples, n_iters
        self.temp, self.gamma = temperature, gamma
        t = lambda x: torch.tensor(x, device=device)
        self.a_lo, self.a_hi, self.a_std = t(A_LO), t(A_HI), t(A_STD)
        self.s_mean, self.s_std = t(norm.s_mean), t(norm.s_std)
        self.a_mean, self.a_std_n = t(norm.a_mean), t(norm.a_std)
        self.reset()

    def reset(self):
        self.mu = torch.zeros(self.H, ACTION_DIM, device=self.device)
        self.mu[:, 6] = 1.0   # suction commanded on by default (pick holds; place releases by sampling)

    @torch.no_grad()
    def _returns(self, state, actions):
        """state: tensor[15] unnormalized. actions: [N,H,7] unnormalized."""
        ee_pos = state[POS][None, None] + torch.cumsum(actions[:, :, 0:3], dim=1)   # [N,H,3]
        z = self.model.encode(((state - self.s_mean) / self.s_std)[None]).repeat(self.N, 1)
        an = (actions - self.a_mean) / self.a_std_n
        ret, disc = torch.zeros(self.N, device=self.device), 1.0
        for t in range(self.H):
            z = self.model.step(z, an[:, t])
            dec = self.model.decode(z)
            sealed_p = torch.sigmoid(dec[:, SEALED])
            fz = dec[:, _FZ] * self.s_std[_FZ] + self.s_mean[_FZ]
            ret = ret + disc * self.reward(ee_pos[:, t], sealed_p, fz)
            disc *= self.gamma
        return ret

    @torch.no_grad()
    def plan(self, state):
        """state: np[15] unnormalized -> action np[7] (suction thresholded)."""
        s = torch.tensor(np.asarray(state, np.float32), device=self.device)
        mu = self.mu
        for _ in range(self.iters):
            eps = torch.randn(self.N, self.H, ACTION_DIM, device=self.device) * self.a_std
            actions = torch.clamp(mu[None] + eps, self.a_lo, self.a_hi)
            ret = self._returns(s, actions)
            w = torch.softmax((ret - ret.max()) / self.temp, dim=0)
            mu = (w[:, None, None] * actions).sum(0)
        self.mu = torch.cat([mu[1:], mu[-1:]], 0)   # warm start for next tick
        a0 = mu[0].clone()
        a0[6] = (a0[6] >= 0.5).float()
        return a0.cpu().numpy()
