"""Transition-state deviation: do failures execute phase transitions at states
outside the demonstrated transition-state distribution?

Two explicit, failure-tied transitions for the pick tasks (no unsupervised
discovery yet — kept tight to the observed failures):

  approach -> descend : where the policy commits to the descent.
      transition-relevant state = lateral alignment + height (ee_x, ee_y, ee_z).
      case_pick failure = descend committed while still laterally short (OOD y).

  contact -> lift     : where the policy starts lifting after contact.
      transition-relevant state = contact force + height (|F|, ee_z).
      battery failure = never lifts / lifts only after pressing past the
      demonstrated lift-onset force (OOD force, or no transition at all).

Detectors run on the recorder/rollout `states.jsonl` (via deviation.load_take),
numpy only.
"""
from __future__ import annotations

import numpy as np

import deviation as dv

# transition-relevant feature indices into the 15-dim state
DESCEND_FEATS = [0, 1, 2]          # ee_x, ee_y, ee_z


def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if len(x) < w:
        return x.astype(float)
    # edge-pad (replicate) so the moving average doesn't dip toward 0 at the
    # ends — a zero-padded 'same' convolution corrupts argmin / the velocity.
    xp = np.pad(x, w // 2, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def detect_descend_onset(states: np.ndarray, v: float = 0.002, k: int = 3):
    """First sustained descent (approach->descend): smoothed z-velocity < -v for
    k consecutive frames. Returns frame index or None."""
    dz = np.gradient(_smooth(states[:, 2]))
    for i in range(1, len(dz) - k):
        if np.all(dz[i:i + k] < -v):
            return i
    return None


def detect_lift_onset(states: np.ndarray, v: float = 0.002, k: int = 3):
    """First sustained lift after the deepest point (contact->lift). Returns
    frame index or None (None = the policy never transitioned to lift)."""
    z = _smooth(states[:, 2])
    dz = np.gradient(z)
    td = int(np.argmin(z))
    for i in range(td + 1, len(dz) - k):
        if np.all(dz[i:i + k] > v):
            return i
    return None


def descend_state(states, i):
    """transition-relevant features at the approach->descend frame."""
    return states[i, DESCEND_FEATS]


def lift_state(states, i):
    """transition-relevant features at the contact->lift frame: |F|, ee_z."""
    return np.array([dv.force_magnitude(states[i:i + 1])[0], states[i, 2]])


def knn_dev(x, ref, k=5):
    """deviation of a single transition state x from the demo cloud ref, z-scored
    on ref. Single-sample wrapper around the k-NN support distance."""
    mean, std = ref.mean(0), ref.std(0)
    std[std < 1e-6] = 1.0
    xn, rn = (x - mean) / std, (ref - mean) / std
    d = np.sqrt(((rn - xn) ** 2).sum(1))
    return float(np.sort(d)[:min(k, len(d))].mean())


def demo_transition_states(phase_dir, which: str):
    """Collect the transition-relevant state at the transition across all demos."""
    rows = []
    for td in dv.list_takes(phase_dir):
        s = dv.load_take(td)["states"]
        if which == "descend":
            i = detect_descend_onset(s)
            if i is not None:
                rows.append(descend_state(s, i))
        elif which == "lift":
            i = detect_lift_onset(s)
            if i is not None:
                rows.append(lift_state(s, i))
    return np.array(rows)
