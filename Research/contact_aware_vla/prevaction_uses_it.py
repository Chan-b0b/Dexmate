#!/usr/bin/env python3
"""Mechanism check (run AFTER training a prev-action model on a checkpoint).

Does the model actually USE the previous action to set descend-vs-lift in mid-air
(i.e., did it learn the phase memory we hoped for)? For mid-air pre-seal frames,
build the 22-dim state with the previous-action slot set to a canonical DESCENDING
vs LIFTING delta (image + all else fixed) and compare predicted dz.

  lifting-prev gives clearly MORE lift than descending-prev  -> model uses the
  previous action as the phase signal (the mid-air ambiguity is resolved).
  no difference -> the model ignores it (history not learned / not useful).

This is the cheap offline proof before committing to an on-robot run.

Run with the vla_venv python (point --checkpoint at the prev-action run):
  /home/dexmate/vla_venv/bin/python Research/contact_aware_vla/prevaction_uses_it.py \
      --checkpoint Research/contact_aware_vla/outputs/smolvla_prevaction/checkpoints/last
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))
sys.path.insert(0, str(HERE.parent / "reactive_chunking"))
from convert_to_lerobot import load_take  # noqa: E402
from run_policy import load_policy, _to_chw  # noqa: E402
from p0_chunk_gap import _load_rgb, _load_depth  # noqa: E402

# canonical previous actions (dpos3, drot3, suction) at realistic magnitudes:
PREV_DESCEND = np.array([0, 0, -0.0018, 0, 0, 0, 1.0], dtype=np.float32)  # was going down
PREV_LIFT = np.array([0, 0, +0.0065, 0, 0, 0, 1.0], dtype=np.float32)     # was going up


def predict_dz(policy, pre, post, state22, rgb, depth, instr, seed=0):
    obs = {"observation.images.head": rgb,
           "observation.state": torch.from_numpy(state22.astype(np.float32)).unsqueeze(0),
           "task": instr, "observation.images.head_depth": depth}
    obs = pre(obs)
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(obs)
    return float(post(chunk[0][0:1]).squeeze(0).cpu().numpy()[2])  # dz (m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--task", default="case_pick")
    ap.add_argument("--recordings", type=Path, default=VLA_TRAIN.parent / "recordings")
    ap.add_argument("--max-takes", type=int, default=6)
    ap.add_argument("--z-lo", type=float, default=0.88)
    ap.add_argument("--z-hi", type=float, default=1.02)
    ap.add_argument("--per-take", type=int, default=6)
    args = ap.parse_args()

    policy, pre, post = load_policy(args.checkpoint)
    sdim = int(policy.config.max_state_dim)
    print(f"checkpoint: {args.checkpoint}\nstate is padded to {sdim}; feeding 22-dim (15 + prev action)\n")

    takes = sorted(p for p in (args.recordings / args.task).iterdir() if p.is_dir())[:args.max_takes]
    dzD, dzL = [], []
    for td in takes:
        L = load_take(td, with_depth=True)
        if L is None:
            continue
        instr, rgb_paths, depth_paths, states, _ = L
        z = states[:, 2]
        seal = states[:, 8] > 0.5
        seal_t = int(np.argmax(seal)) if seal.any() else len(states)
        cand = [i for i in range(seal_t) if args.z_lo < z[i] < args.z_hi]  # mid-air, pre-seal
        if not cand:
            continue
        for i in np.unique(np.linspace(cand[0], cand[-1], min(args.per_take, len(cand))).astype(int)):
            rgb, dep = _to_chw(_load_rgb(rgb_paths[i])), _to_chw(_load_depth(depth_paths[i]))
            sD = np.concatenate([states[i], PREV_DESCEND])
            sL = np.concatenate([states[i], PREV_LIFT])
            dzD.append(predict_dz(policy, pre, post, sD, rgb, dep, instr))
            dzL.append(predict_dz(policy, pre, post, sL, rgb, dep, instr))

    dzD, dzL = np.array(dzD), np.array(dzL)
    print(f"mid-air pre-seal frames probed: {len(dzD)}")
    print(f"  predicted dz | prev=DESCEND: {dzD.mean()*1000:+.2f} mm   prev=LIFT: {dzL.mean()*1000:+.2f} mm")
    print(f"  gap (LIFT - DESCEND) = {(dzL-dzD).mean()*1000:+.2f} mm   "
          f"(fraction of frames where LIFT>DESCEND: {(dzL>dzD).mean()*100:.0f}%)")
    gap = (dzL - dzD).mean() * 1000
    print("\n  => 모델이 previous action으로 phase를 판별함 (history learned)" if gap > 0.5 else
          "\n  => previous action 영향 미미 (history not used) — 가설 미성립 가능")


if __name__ == "__main__":
    main()
