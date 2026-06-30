#!/usr/bin/env python3
"""Closed-loop drift analyzer (P0c).

Takes a rollout.npz from cl_drift_record.py (the policy's OWN closed-loop
trajectory) and measures the open-loop chunk gap on it with the EXACT same
metric P0 used on expert replay — same fixed-noise method, so the only
difference is the observation source: drifted closed-loop obs vs expert obs.
Then prints the two side by side.

It recomputes both the stale and the fresh chunk from the logged observations
(deploy = C[s][t-s], fresh = C[t][0]); it does NOT use the live random-noise
action, so divergence reflects observation drift, not flow-matching sampling.
The hypothesis P0 could not test: on the drifted rollout, deploy-vs-fresh
divergence is larger (and rises near contact) because the observation left the
expert manifold.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/reactive_chunking/cl_drift_analyze.py \
      --rollout Research/reactive_chunking/rollouts/run1/rollout.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))
sys.path.insert(0, str(HERE))           # reuse P0's metric functions
from run_policy import load_policy, _to_chw  # noqa: E402
from convert_to_lerobot import quat_mul, quat_conj, quat_to_rotvec  # noqa: E402
import p0_chunk_gap as p0  # noqa: E402


def predict_chunks_inmem(policy, pre, post, states, images, depths, instr, seed):
    """Fresh action chunk per tick from logged (in-memory) observations.
    Deterministic (fixed seed) — identical method to p0.predict_chunks."""
    chunks = []
    for i in range(len(states)):
        obs = {
            "observation.images.head": _to_chw(images[i]),
            "observation.state": torch.from_numpy(states[i].astype(np.float32)).unsqueeze(0),
            "task": str(instr[i]),
        }
        if depths is not None:
            obs["observation.images.head_depth"] = _to_chw(depths[i])
        obs = pre(obs)
        policy.reset()
        torch.manual_seed(seed)
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(obs)[0]
        chunks.append(np.stack([post(chunk[k:k + 1]).squeeze(0).cpu().numpy()
                                for k in range(chunk.shape[0])]))
        if (i + 1) % 25 == 0:
            print(f"  tick {i + 1}/{len(states)}")
    return np.asarray(chunks)


def realized_actions(states):
    """What the robot actually did next, in the converter's action convention
    (delta pos + rotvec(R_{t+1}R_t^T) + next suction) — the rollout's own truth."""
    pos, quat, suc = states[:, :3], states[:, 3:7], states[:, 7]
    dpos = pos[1:] - pos[:-1]
    drot = np.stack([quat_to_rotvec(quat_mul(quat[i + 1], quat_conj(quat[i])))
                     for i in range(len(states) - 1)])
    return np.concatenate([dpos, drot, suc[1:, None]], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_TRAIN / "outputs/smolvla_depthseal_fixed/checkpoints/last")
    ap.add_argument("--expert-summary", type=Path, default=HERE / "p0_results/summary.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.rollout.parent
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(args.rollout, allow_pickle=True)
    states, images, instr = d["state"], d["image"], d["instruction"]
    depths = d["depth"] if "depth" in d.files else None
    n = len(states)

    policy, pre, post = load_policy(args.checkpoint)
    K = int(policy.config.chunk_size)
    print(f"rollout {args.rollout}: {n} ticks, K={K}, depth={'yes' if depths is not None else 'no'}\n")

    chunks = predict_chunks_inmem(policy, pre, post, states, images, depths, instr, args.seed)

    # segment by instruction change; deploy re-grounds from each task's first tick
    seg = [0] + [i for i in range(1, n) if instr[i] != instr[i - 1]] + [n]
    all_rows, all_stale, all_lags = [], [], []
    for a, b in zip(seg[:-1], seg[1:]):
        c, st = chunks[a:b], states[a:b]
        acts = realized_actions(st)
        rows, stale, lags, events = p0.analyze_take(c, acts, st, K)
        print(f"  task '{instr[a]}' [{a}:{b}]: seal events {events}, lags {lags}")
        all_rows += rows; all_stale += stale; all_lags += lags

    summary = p0.summarize(all_rows, all_stale, all_lags, K, out, make_plots=True)

    if args.expert_summary.exists():
        exp = json.loads(args.expert_summary.read_text())
        e, c = exp["deploy_vs_fresh_dpos_mm"], summary["deploy_vs_fresh_dpos_mm"]

        def _fmt(v):
            return "  n/a" if v is None else f"{v:.3f}"
        print("\n----- CLOSED-LOOP vs EXPERT REPLAY  (deploy-vs-fresh dpos, mm) -----")
        print(f"  near contact : closed-loop {_fmt(c['near_contact'])}  | expert {_fmt(e['near_contact'])}")
        print(f"  far          : closed-loop {_fmt(c['far_from_contact'])}  | expert {_fmt(e['far_from_contact'])}")
        print("  (if closed-loop >> expert, especially near contact, drift makes "
              "re-grounding matter — the gap P0 could not see.)")


if __name__ == "__main__":
    main()
