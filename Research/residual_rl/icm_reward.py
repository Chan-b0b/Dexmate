#!/usr/bin/env python
"""Online reversed-curiosity reward, scoring live transitions with a pretrained ICM.

Loads an icm_0708_<variant>.pt checkpoint (see
LGES/vla_training/smolvla_meanflow/scripts/train_icm.py, uploaded to HF as
Chanho-Lee/icm_case_pick_0708) and scores (state, action, next_state) transitions the
same way training did: r = exp(-eta * ensemble prediction error), high when the
transition stays on the demo manifold.

variant="proprio" (the actor's default) needs no vision feature -- the only latency-safe
choice inside the 15 Hz control loop. "vision"/"both" need a mean-pooled vision feature
vector per step (960-d/camera, see extract_icm_features.py) -- NOT wired up here; the
caller would have to reuse the base policy's own vision tower to avoid a second image
encode, which actor.py does not currently do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SMOLVLA_MEANFLOW = Path(__file__).resolve().parents[2] / "LGES" / "vla_training" / "smolvla_meanflow"
if str(_SMOLVLA_MEANFLOW) not in sys.path:
    sys.path.insert(0, str(_SMOLVLA_MEANFLOW))
from smolvla_meanflow.icm import ICMEnsemble  # noqa: E402


# Live-vs-training wrench baseline offset, measured 2026-07-16: mean(actor.py rollout
# state[9:15]) - mean(lges_case_pick_0708 observation.state[9:15]), order fx..tz.
# Live position/orientation matched the demo distribution within 1 sigma while 5 of 6
# wrench dims sat 2.5-4.8 sigma off -- consistent with F/T sensor baseline drift since
# the 0708 demos. Subtracting it restored rollout reward from ~0.17 to ~0.95 median
# (vs held-out demo median 0.98). STOPGAP: if the sensor is ever re-zeroed or the ICM
# retrained on fresh demos, re-measure or drop this.
WRENCH_OFFSET_0716 = np.array([3.1942, 2.2735, -10.1769, 0.2337, -0.5755, -0.3922],
                              dtype=np.float32)


def _resolve_checkpoint(checkpoint: str | Path, filename: str) -> Path:
    """`checkpoint` may be a direct file path, a local directory containing `filename`,
    or an HF Hub repo id (e.g. "Chanho-Lee/icm_case_pick_0708") -- same three-way
    resolution run_policy.py already uses for policy/FiLM checkpoints."""
    path = Path(checkpoint)
    if path.is_file():
        return path
    if path.is_dir():
        return path / filename
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=str(checkpoint), filename=filename))


class ICMRewarder:
    def __init__(self, checkpoint: str | Path, filename: str = "icm_0708_proprio.pt",
                 device: str = "cpu"):
        resolved = _resolve_checkpoint(checkpoint, filename)
        self.icm = ICMEnsemble.load(resolved, map_location=device).to(device)
        self.icm.eval()
        self.variant = self.icm.variant
        self.device = device
        if self.variant != "proprio":
            print(f"[icm_reward] WARNING: variant={self.variant!r} needs a vision "
                  f"feature per step; pass it explicitly to reward() or this will "
                  f"raise on the first call.")

    @staticmethod
    def _correct_wrench(state: np.ndarray, wrench_offset: np.ndarray | None) -> np.ndarray:
        """Subtract a measured live-vs-training wrench offset from state[9:15]
        (fx,fy,fz,tx,ty,tz) before scoring -- a stopgap for a suspected F/T sensor
        baseline drift since the case_pick_0708 demos were recorded (live
        position/orientation match the training distribution; wrench alone doesn't,
        see the residual_rl reward-diagnosis discussion). NOT a permanent fix --
        re-zeroing the sensor or recalibrating the ICM on freshly-recorded demos
        would be. None (default) applies no correction."""
        if wrench_offset is None:
            return state
        corrected = np.array(state, dtype=np.float32, copy=True)
        corrected[9:15] -= wrench_offset
        return corrected

    def _feat(self, state: np.ndarray, vision: np.ndarray | None) -> torch.Tensor:
        if self.variant == "proprio":
            f = state
        else:
            if vision is None:
                raise ValueError(f"variant={self.variant!r} requires a vision feature vector")
            f = vision if self.variant == "vision" else np.concatenate([state, vision])
        return torch.as_tensor(f, dtype=torch.float32, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        vision: np.ndarray | None = None,
        next_vision: np.ndarray | None = None,
        wrench_offset: np.ndarray | None = None,
    ) -> float:
        """`action` must match the ICM's action_dim -- the dataset's FULL recorded
        action (e.g. 7-d dpos+drot+suction for the delta case_pick checkpoints), not
        just the 6-d residual pose. Pass the actually-EXECUTED (post-clamp) action, the
        same convention train_icm.py used (the dataset's recorded action, not the raw
        policy prediction). `wrench_offset`: see _correct_wrench()."""
        feat = self._feat(self._correct_wrench(state, wrench_offset), vision)
        next_feat = self._feat(self._correct_wrench(next_state, wrench_offset), next_vision)
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.icm.reward(feat, act, next_feat).squeeze(0))

    @torch.no_grad()
    def feature_residual(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        vision: np.ndarray | None = None,
        next_vision: np.ndarray | None = None,
        wrench_offset: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Alternative, un-calibrated OOD signal: raw feature-space residual
        next_feature - predicted_next_feature (ensemble-mean), in the SAME units as
        `state` -- unlike reward()'s exp(-eta*error), which is calibrated to
        case_pick's OWN held-out demo error scale (p95 -> r=0.8) and can saturate
        near 0 for a structurally different task/pose region, hiding relative
        differences that the raw residual still shows. `wrench_offset`: see
        _correct_wrench(). Returns (||residual||, residual)."""
        feat = self._feat(self._correct_wrench(state, wrench_offset), vision)
        next_feat = self._feat(self._correct_wrench(next_state, wrench_offset), next_vision)
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        pred_delta_norm = self.icm.predict(feat, act).mean(dim=0)  # ensemble-mean, normalized
        pred_next_feat = feat + pred_delta_norm * self.icm.feat_std
        residual = (next_feat - pred_next_feat).squeeze(0).cpu().numpy()
        return float(np.linalg.norm(residual)), residual
