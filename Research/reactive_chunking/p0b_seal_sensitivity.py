#!/usr/bin/env python3
"""P0b — is the policy's action causally sensitive to the seal bit?

P0 showed the stale chunk barely diverges from a fresh re-grounding on expert
data, and never disagrees on suction near contact. That raises the linchpin
question for the whole "fast contact-reactive head" idea: does SmolVLA's action
even DEPEND on vacuum_sealed? If flipping the seal bit doesn't move the action,
re-grounding on a seal event is pointless regardless of latency.

Method (single-feature causal intervention, do-operator on the state): for
frames around the seal event, predict the action with the seal bit forced to 0
vs forced to 1, holding the image and all other state fixed. The effect size is
||a(seal=1) - a(seal=0)||, compared to the natural per-step action magnitude.
For contrast we do the same for the echoed suction-command bit (the classic
causal-confusion nuisance feature).

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/reactive_chunking/p0b_seal_sensitivity.py \
      [--max-takes 6] [--window-pre 30] [--window-post 10]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))

import cv2  # noqa: E402
from convert_to_lerobot import load_take, colorize_depth, IMG_W, IMG_H  # noqa: E402
from run_policy import load_policy, _to_chw  # noqa: E402

SEAL_COL, SUCTION_COL = 8, 7  # state layout (convert_to_lerobot.STATE_NAMES)


def _load_rgb(p):
    img = cv2.imread(str(p))
    return cv2.cvtColor(cv2.resize(img, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)


def _load_depth(p):
    return colorize_depth(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))


def first_action(policy, pre, post, state, rgb_t, depth_t, instruction, seed):
    """First action of the predicted chunk for a given (possibly edited) state."""
    obs = {
        "observation.images.head": rgb_t,
        "observation.images.head_depth": depth_t,
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": instruction,
    }
    obs = pre(obs)
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(obs)
    return post(chunk[0][0:1]).squeeze(0).cpu().numpy()


def seal_rising_edges(states):
    s = states[:, SEAL_COL] > 0.5
    return [i for i in range(1, len(s)) if s[i] and not s[i - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="case_pick")
    ap.add_argument("--recordings", type=Path, default=VLA_TRAIN.parent / "recordings")
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_TRAIN / "outputs/smolvla_depthseal_fixed/checkpoints/last")
    ap.add_argument("--max-takes", type=int, default=6)
    ap.add_argument("--window-pre", type=int, default=30)
    ap.add_argument("--window-post", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "p0_results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy, pre, post = load_policy(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}\n")

    takes = sorted(p for p in (args.recordings / args.task).iterdir() if p.is_dir())[:args.max_takes]
    seal_eff, suc_eff, seal_dz, seal_dsuc, nat_step = [], [], [], [], []
    for ti, td in enumerate(takes):
        loaded = load_take(td, with_depth=True)
        if loaded is None:
            continue
        instruction, rgb_paths, depth_paths, states, actions = loaded
        events = seal_rising_edges(states)
        if not events:
            continue
        e = events[0]
        lo, hi = max(0, e - args.window_pre), min(len(states), e + args.window_post)
        nat_step.append(np.linalg.norm(actions[:, :3], axis=1).mean())
        print(f"  [{ti+1}/{len(takes)}] {td.name}: seal@{e}, frames {lo}..{hi}")
        for i in range(lo, hi):
            rgb_t, depth_t = _to_chw(_load_rgb(rgb_paths[i])), _to_chw(_load_depth(depth_paths[i]))
            s = states[i]
            s0, s1 = s.copy(), s.copy(); s0[SEAL_COL], s1[SEAL_COL] = 0.0, 1.0
            a_s0 = first_action(policy, pre, post, s0, rgb_t, depth_t, instruction, args.seed)
            a_s1 = first_action(policy, pre, post, s1, rgb_t, depth_t, instruction, args.seed)
            seal_eff.append(np.linalg.norm(a_s1[:3] - a_s0[:3]))
            seal_dz.append(a_s1[2] - a_s0[2])
            seal_dsuc.append(a_s1[6] - a_s0[6])
            # contrast: echoed suction-command bit (causal-confusion candidate)
            c0, c1 = s.copy(), s.copy(); c0[SUCTION_COL], c1[SUCTION_COL] = 0.0, 1.0
            a_c0 = first_action(policy, pre, post, c0, rgb_t, depth_t, instruction, args.seed)
            a_c1 = first_action(policy, pre, post, c1, rgb_t, depth_t, instruction, args.seed)
            suc_eff.append(np.linalg.norm(a_c1[:3] - a_c0[:3]))

    seal_eff, suc_eff = np.array(seal_eff), np.array(suc_eff)
    nat = float(np.mean(nat_step))
    summary = {
        "frames_probed": int(len(seal_eff)),
        "natural_per_step_dpos_mm": nat * 1000,
        "seal_intervention": {
            "dpos_effect_mm": float(seal_eff.mean() * 1000),
            "dpos_effect_frac_of_step": float(seal_eff.mean() / nat),
            "dz_effect_mm_mean": float(np.mean(seal_dz) * 1000),
            "dsuction_effect_mean": float(np.mean(seal_dsuc)),
        },
        "suction_bit_intervention": {
            "dpos_effect_mm": float(suc_eff.mean() * 1000),
            "dpos_effect_frac_of_step": float(suc_eff.mean() / nat),
        },
    }
    (args.out / "seal_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print("\n===== P0b SEAL SENSITIVITY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out/'seal_sensitivity.json'}")


if __name__ == "__main__":
    main()
