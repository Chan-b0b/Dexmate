"""Deviation-from-demonstration profiling for closed-loop rollouts.

Measures, per timestep, how far a rollout's robot state is from the nearest
demonstrated states (k-NN distance in normalised state space). Splitting the
15-dim state into feature groups separates *gradual pose drift* (mode i,
covariate shift of the conditioning state) from *contact events* (force
spikes, mode ii).

Data source: recorder `states.jsonl` takes. The expert demos
(LGES/recordings/<phase>/) and closed-loop rollouts (e.g.
Research/intervention/interventions/...) share this schema, so the reference
manifold and the rollout are reconstructed by the same path. numpy only — no
parquet / lerobot dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 15-dim observation.state layout, matching the lerobot dataset
# (LGES/vla_training/datasets/lges_suction/meta/info.json).
STATE_NAMES = [
    "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
    "suction", "vacuum_sealed", "fx", "fy", "fz", "tx", "ty", "tz",
]
FEATURE_GROUPS = {
    "pose":    [0, 1, 2, 3, 4, 5, 6],     # EE position + orientation -> drift / mode (i)
    "ee_pos":  [0, 1, 2],
    "ee_quat": [3, 4, 5, 6],
    "force":   [9, 10, 11, 12, 13, 14],   # wrench -> contact / mode (ii) trigger
    "full":    list(range(15)),
}


def frame_to_state(f: dict) -> np.ndarray:
    """Reconstruct the 15-dim observation.state from one states.jsonl frame."""
    ee, w = f["ee"], f["wrench"]
    return np.array([
        *ee["pos"], *ee["quat_wxyz"],
        1.0 if f.get("suction_cmd") else 0.0,
        1.0 if f.get("vacuum_sealed") else 0.0,
        w["fx"], w["fy"], w["fz"], w["tx"], w["ty"], w["tz"],
    ], dtype=np.float64)


def _canon_quat(states: np.ndarray) -> np.ndarray:
    """Flip quaternion sign so qw>=0 (quaternions double-cover SO(3))."""
    s = states.copy()
    flip = s[:, 3] < 0
    s[flip, 3:7] *= -1.0
    return s


def load_take(take_dir) -> dict:
    """Load a recorder take: 15-dim states (T,15), timestamps, phase, meta."""
    take_dir = Path(take_dir)
    frames = [json.loads(l) for l in (take_dir / "states.jsonl").open() if l.strip()]
    states = _canon_quat(np.stack([frame_to_state(f) for f in frames]))
    t = np.array([f["t"] for f in frames], dtype=np.float64)
    meta = json.loads((take_dir / "meta.json").read_text())
    return {
        "dir": take_dir, "name": take_dir.name, "phase": meta.get("phase"),
        "states": states, "t": t, "meta": meta,
    }


def list_takes(phase_dir) -> list[Path]:
    """All take dirs under a phase dir that contain a states.jsonl."""
    return sorted(p for p in Path(phase_dir).iterdir()
                  if p.is_dir() and (p / "states.jsonl").exists())


class Normalizer:
    """Per-dimension z-score fit on the reference (demo) states."""

    def __init__(self, ref: np.ndarray):
        self.mean = ref.mean(0)
        std = ref.std(0)
        # A dim that is constant across demos (e.g. a flag never toggled) gets
        # std=1 so any rollout difference contributes its raw value rather than
        # exploding the distance.
        std[std < 1e-6] = 1.0
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def knn_deviation(x_norm: np.ndarray, ref_norm: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance from each row of x to its k nearest rows in ref.

    This is a support-distance: how far the rollout state is from the closest
    demonstrated states. Distance to the *manifold* of demos, not to their mean
    (demos form curved trajectories, not a Gaussian blob), so k-NN, not
    Mahalanobis.
    """
    d2 = ((x_norm ** 2).sum(1)[:, None]
          + (ref_norm ** 2).sum(1)[None, :]
          - 2 * x_norm @ ref_norm.T)
    np.maximum(d2, 0, out=d2)
    k = min(k, ref_norm.shape[0])
    nearest = np.partition(d2, k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(1)


def deviation_profile(states: np.ndarray, ref_states: np.ndarray,
                      norm: Normalizer, groups, k: int = 5) -> dict:
    """Per-frame k-NN deviation for each feature group."""
    xn, rn = norm(states), norm(ref_states)
    return {g: knn_deviation(xn[:, FEATURE_GROUPS[g]], rn[:, FEATURE_GROUPS[g]], k)
            for g in groups}


def build_reference(phase_dir, exclude_name: str | None = None) -> np.ndarray:
    """Concatenated demo states for a phase, optionally leaving one take out."""
    blocks = []
    for tk in list_takes(phase_dir):
        if exclude_name and tk.name == exclude_name:
            continue
        blocks.append(load_take(tk)["states"])
    if not blocks:
        raise FileNotFoundError(f"no demo takes found under {phase_dir}")
    return np.concatenate(blocks, 0)


def loo_baseline(phase_dir, groups, k: int = 5) -> dict:
    """Leave-one-out: every demo scored against the other demos of its phase.

    Returns the pooled in-distribution per-frame deviations per group — the
    band a rollout curve should be read against.
    """
    pooled = {g: [] for g in groups}
    for tk in list_takes(phase_dir):
        ref = build_reference(phase_dir, exclude_name=tk.name)
        prof = deviation_profile(load_take(tk)["states"], ref, Normalizer(ref), groups, k)
        for g in groups:
            pooled[g].append(prof[g])
    return {g: np.concatenate(v) for g, v in pooled.items()}


def force_magnitude(states: np.ndarray) -> np.ndarray:
    """|F| (N) from the wrench force components — contact onset marker."""
    return np.linalg.norm(states[:, 9:12], axis=1)
