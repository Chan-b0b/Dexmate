#!/usr/bin/env python3
"""Analytic reward for LGES deploy-time MPC.

Per-phase contact targets are DERIVED FROM DATA — the median recorded EE pose at
the seal/release toggle — so they live in the exact frame the dynamics model
predicts in (no taught-pose frame conversion). The reward shapes the objective:

  pick : reach the contact pose -> achieve seal (model-predicted) -> lift to hover
  place: reach the slot while holding -> release at the slot -> lift to hover

plus a soft penalty on excess contact force. Orientation is not rewarded (the
suction cup is held vertical throughout).
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# task_index order matches convert_to_lerobot.SUCTION_TASKS (verified against the
# data: contact poses line up with the taught poses per phase).
PHASE_KIND = {"case_pick": "pick", "case_place": "place",
              "battery_1_pick": "pick", "battery_1_place": "place",
              "battery_2_pick": "pick", "battery_2_place": "place"}


def compute_targets(dataset_root):
    """{phase: {contact[3], hover_z, kind}} from the training takes."""
    cols = ["observation.state", "episode_index", "frame_index", "task_index"]
    files = sorted(glob.glob(str(Path(dataset_root) / "data" / "chunk-*" / "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    phases = list(PHASE_KIND)
    targets = {}
    for ti, phase in enumerate(phases):
        sub = df[df.task_index == ti]
        contacts, hovers = [], []
        for _, g in sub.groupby("episode_index"):
            S = np.stack(g.sort_values("frame_index")["observation.state"].values)
            ch = np.where(S[1:, 8] != S[:-1, 8])[0]   # first seal toggle
            if len(ch):
                contacts.append(S[ch[0] + 1, :3])
            hovers.append(S[-1, 2])                     # final (lifted) EE z
        targets[phase] = {"contact": np.median(contacts, 0).astype(np.float32),
                          "hover_z": float(np.median(hovers)),
                          "kind": PHASE_KIND[phase]}
    return targets


class PhaseReward:
    """Dense reward r(ee_pos, sealed_prob, fz) for one sub-task phase.

    Works on arbitrarily-batched tensors. The model supplies sealed_prob and fz;
    ee_pos comes from analytic integration of the action deltas."""

    def __init__(self, target, w_seal=8.0, w_reach=6.0,
                 w_force=0.05, f_safe=20.0, near_sigma=0.05):
        self.contact = torch.as_tensor(target["contact"])
        self.hover_z = target["hover_z"]
        self.kind = target["kind"]
        self.w_seal, self.w_reach = w_seal, w_reach
        self.w_force, self.f_safe, self.near_sigma = w_force, f_safe, near_sigma

    def to(self, device):
        self.contact = self.contact.to(device)
        return self

    def __call__(self, ee_pos, sealed_prob, fz):
        # Reach is ALWAYS active so descent can't be switched off by the model's
        # own seal prediction. The seal/release bonus is GATED by proximity to
        # the contact pose — a seal in mid-air is physically meaningless, so the
        # planner cannot reward-hack it without first descending to the object.
        d = torch.linalg.norm(ee_pos - self.contact, dim=-1)
        near = torch.exp(-(d ** 2) / (self.near_sigma ** 2))
        reach = -self.w_reach * d
        force_pen = -self.w_force * torch.relu(fz.abs() - self.f_safe)
        if self.kind == "pick":
            event = self.w_seal * sealed_prob * near          # seal at the object
        else:
            event = self.w_seal * (1.0 - sealed_prob) * near  # release at the slot
        return reach + event + force_pen
