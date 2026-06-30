#!/usr/bin/env python3
"""Dataset + normalization for the LGES latent dynamics model.

Loads ONLY the low-dim columns (observation.state[15], action[7]) from the
LeRobotDataset parquet files — images live in separate files and are never
touched, so loading is fast. Builds fixed-length sub-trajectory windows
(states[H+1,15], actions[H,7]) for multi-step world-model training, plus the
normalization stats the model needs.

State layout (matches convert_to_lerobot.py STATE_NAMES):
  [0:3]  ee pos (m)          [3:7]  ee quat wxyz (unit, sign-continuous)
  [7]    suction cmd {0,1}   [8]    vacuum_sealed {0,1}  (physical DI0 seal)
  [9:15] raw wrench fx..tz (N, N*m)
Action layout (ACTION_NAMES): [0:3] dpos(m) [3:6] drot rotvec(rad) [6] suction[t+1]

By construction pos[t+1]=pos[t]+dpos, quat[t+1]=drot∘quat[t] and
suction[t+1]=action[6] are EXACT. The only genuinely-learned dynamics are
wrench[t+1] and sealed[t+1] (contact forces + the suction-grab event) — that is
what the validation harness scores hardest.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# state-vector index map
POS = slice(0, 3)
QUAT = slice(3, 7)
SUCTION = 7
SEALED = 8
WRENCH = slice(9, 15)
STATE_DIM = 15
ACTION_DIM = 7
# continuous state dims that get z-scored; quat(3:7), suction(7), sealed(8) are
# left raw (quat is unit; the two flags are {0,1}).
_ZSCORE_STATE = list(range(0, 3)) + list(range(9, 15))
_ZSCORE_ACTION = list(range(0, 6))  # dpos+drot; suction action(6) left raw

_COLS = ["observation.state", "action", "episode_index", "frame_index"]


def load_episodes(root):
    """Return [(states[K,15], actions[K,7]), ...] per episode, frame-ordered.

    Within a dataset episode, action[i] maps state[i] -> state[i+1]; the last
    frame's action targets a state dropped during conversion, so windows never
    use it (see WindowDataset)."""
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}/data/chunk-*/")
    df = pd.concat([pd.read_parquet(f, columns=_COLS) for f in files], ignore_index=True)
    episodes = []
    for _, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        s = np.stack(g["observation.state"].values).astype(np.float32)
        a = np.stack(g["action"].values).astype(np.float32)
        episodes.append((s, a))
    return episodes


class Normalizer:
    """(x - mean) / std with mean=0, std=1 on the dims we keep raw."""

    def __init__(self, s_mean, s_std, a_mean, a_std):
        self.s_mean = np.asarray(s_mean, np.float32)
        self.s_std = np.asarray(s_std, np.float32)
        self.a_mean = np.asarray(a_mean, np.float32)
        self.a_std = np.asarray(a_std, np.float32)

    @classmethod
    def fit(cls, episodes):
        S = np.concatenate([s for s, _ in episodes], 0)
        A = np.concatenate([a for _, a in episodes], 0)
        s_mean = np.zeros(STATE_DIM, np.float32)
        s_std = np.ones(STATE_DIM, np.float32)
        s_mean[_ZSCORE_STATE] = S[:, _ZSCORE_STATE].mean(0)
        s_std[_ZSCORE_STATE] = S[:, _ZSCORE_STATE].std(0) + 1e-6
        a_mean = np.zeros(ACTION_DIM, np.float32)
        a_std = np.ones(ACTION_DIM, np.float32)
        a_mean[_ZSCORE_ACTION] = A[:, _ZSCORE_ACTION].mean(0)
        a_std[_ZSCORE_ACTION] = A[:, _ZSCORE_ACTION].std(0) + 1e-6
        return cls(s_mean, s_std, a_mean, a_std)

    def norm_state(self, s):
        return (s - self.s_mean) / self.s_std

    def denorm_state(self, s):
        return s * self.s_std + self.s_mean

    def norm_action(self, a):
        return (a - self.a_mean) / self.a_std

    def to_dict(self):
        return {k: getattr(self, k).tolist()
                for k in ("s_mean", "s_std", "a_mean", "a_std")}

    @classmethod
    def from_dict(cls, d):
        return cls(d["s_mean"], d["s_std"], d["a_mean"], d["a_std"])


class WindowDataset(Dataset):
    """Fixed-length sub-trajectories for multi-step rollout training.

    Each item: (states[H+1,15] normalized, actions[H,7] normalized). actions[t]
    steps states[t] -> states[t+1]."""

    def __init__(self, episodes, horizon, norm: Normalizer):
        self.h = horizon
        self.eps = []          # normalized (states, actions) per episode
        self.windows = []      # (ep_idx, start)
        for s, a in episodes:
            K = len(s)
            if K <= horizon:
                continue
            self.eps.append((norm.norm_state(s), norm.norm_action(a)))
            ei = len(self.eps) - 1
            for t in range(0, K - horizon):  # t..t+H states valid, actions t..t+H-1
                self.windows.append((ei, t))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ei, t = self.windows[i]
        s, a = self.eps[ei]
        h = self.h
        return (torch.from_numpy(s[t:t + h + 1]).clone(),
                torch.from_numpy(a[t:t + h]).clone())
