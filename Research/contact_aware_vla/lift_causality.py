#!/usr/bin/env python3
"""P2 — is the grasp->lift transition seal-GATED or open-loop TIMING-driven?

The thesis from the corrected P1/direct-test finding: the policy lifts after a
grasp on a learned dwell-then-lift timing pattern, NOT because it senses the
seal. This probe tests that in-distribution, across all pick tasks, two ways:

  Q1 (at-seal effect): around the real seal rising-edge, flip the seal bit 0<->1
     and measure |dpos|. Expect SMALL (the immediate seal doesn't jolt the action).

  Q2 (lift gating, the crux): at frames where the arm is actually LIFTING
     post-seal (recorded z rising), predict dz with seal forced to 1 vs forced to
     0, holding image + all other state fixed.
       - seal-GATED  => seal=0 kills the lift (dz_0 << dz_1)
       - TIMING-driven => seal=0 lifts just as much (dz_0 ~= dz_1)
     Compared against the recorded lift dz for scale.

Counterfactual by design (that's what a do-intervention is), but kept at
IN-DISTRIBUTION operating points (real lift frames), unlike the global-swing
P1 which manufactured OOD inputs and inflated the apparent sensitivity.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/contact_aware_vla/lift_causality.py \
      [--tasks case_pick battery_1_pick battery_2_pick] [--max-takes 6]
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
sys.path.insert(0, str(HERE.parent / "reactive_chunking"))
from convert_to_lerobot import load_take  # noqa: E402
from run_policy import load_policy, _to_chw  # noqa: E402
from p0_chunk_gap import _load_rgb, _load_depth  # noqa: E402

SEAL_COL = 8
LIFT_DZ_M = 0.0005  # recorded z must rise >0.5mm/frame to count as "lifting"


def first_action(policy, pre, post, state, rgb, depth, instr, seed=0):
    obs = {"observation.images.head": rgb,
           "observation.state": torch.from_numpy(state.astype(np.float32)).unsqueeze(0),
           "task": instr, "observation.images.head_depth": depth}
    obs = pre(obs)
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode():
        ch = policy.predict_action_chunk(obs)
    return post(ch[0][0:1]).squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+",
                    default=["case_pick", "battery_1_pick", "battery_2_pick"])
    ap.add_argument("--recordings", type=Path, default=VLA_TRAIN.parent / "recordings")
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_TRAIN / "outputs/smolvla_depthseal_fixed/checkpoints/last")
    ap.add_argument("--max-takes", type=int, default=6)
    ap.add_argument("--lift-frames", type=int, default=8)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy, pre, post = load_policy(args.checkpoint)
    res = {}
    for task in args.tasks:
        td_all = sorted(p for p in (args.recordings / task).iterdir() if p.is_dir())[:args.max_takes]
        trans_dpos, lift = [], []   # lift: (dz0, dz1, rec_dz) per frame, mm
        for td in td_all:
            L = load_take(td, with_depth=True)
            if L is None:
                continue
            instr, rgb_paths, depth_paths, states, _ = L
            z = states[:, 2]
            seal = states[:, SEAL_COL] > 0.5
            edges = [i for i in range(1, len(seal)) if seal[i] and not seal[i - 1]]
            if not edges:
                continue
            e = edges[0]

            def probe(i):
                rgb, dep = _to_chw(_load_rgb(rgb_paths[i])), _to_chw(_load_depth(depth_paths[i]))
                s0, s1 = states[i].copy(), states[i].copy()
                s0[SEAL_COL], s1[SEAL_COL] = 0.0, 1.0
                return (first_action(policy, pre, post, s0, rgb, dep, instr),
                        first_action(policy, pre, post, s1, rgb, dep, instr))

            # Q1: at-seal transition window
            for i in range(max(0, e - 3), min(len(states), e + 4)):
                a0, a1 = probe(i)
                trans_dpos.append(float(np.linalg.norm(a1[:3] - a0[:3])))
            # Q2: real lifting frames post-seal (recorded z rising)
            lf = [i for i in range(e, len(states) - 1) if z[i + 1] - z[i] > LIFT_DZ_M]
            if lf:
                idxs = np.unique(np.linspace(lf[0], lf[-1], min(args.lift_frames, len(lf))).astype(int))
                for i in idxs:
                    a0, a1 = probe(i)
                    lift.append((a0[2] * 1000, a1[2] * 1000, (z[i + 1] - z[i]) * 1000))

        lift = np.array(lift) if lift else np.zeros((0, 3))
        d = {
            "takes": len(td_all),
            "at_seal_dpos_mm": float(np.mean(trans_dpos) * 1000) if trans_dpos else None,
            "lift_frames": int(len(lift)),
            "pred_dz_seal0_mm": float(lift[:, 0].mean()) if len(lift) else None,
            "pred_dz_seal1_mm": float(lift[:, 1].mean()) if len(lift) else None,
            "recorded_dz_mm": float(lift[:, 2].mean()) if len(lift) else None,
            "lift_gain_from_seal_mm": float((lift[:, 1] - lift[:, 0]).mean()) if len(lift) else None,
        }
        if len(lift) and d["recorded_dz_mm"]:
            d["seal_gating_ratio"] = d["lift_gain_from_seal_mm"] / d["recorded_dz_mm"]
        res[task] = d
        print(f"\n[{task}] takes={d['takes']} | at-seal dpos={d['at_seal_dpos_mm']:.3f}mm "
              f"| lift frames={d['lift_frames']}")
        if len(lift):
            print(f"   pred lift dz: seal=0 {d['pred_dz_seal0_mm']:+.2f}mm  seal=1 {d['pred_dz_seal1_mm']:+.2f}mm "
                  f"| recorded {d['recorded_dz_mm']:+.2f}mm")
            print(f"   lift gain from seal = {d['lift_gain_from_seal_mm']:+.3f}mm "
                  f"({d['seal_gating_ratio']*100:+.0f}% of recorded lift)")

    (args.out / "lift_causality.json").write_text(json.dumps(res, indent=2))
    # verdict
    gains = [d["lift_gain_from_seal_mm"] for d in res.values() if d.get("lift_gain_from_seal_mm") is not None]
    recs = [d["recorded_dz_mm"] for d in res.values() if d.get("recorded_dz_mm")]
    if gains and recs:
        ratio = np.mean(gains) / np.mean(recs)
        print("\n===== VERDICT =====")
        print(f"avg lift gain from seal = {np.mean(gains):+.3f}mm vs recorded lift {np.mean(recs):+.2f}mm "
              f"({ratio*100:+.0f}%)")
        print("  TIMING-DRIVEN (thesis holds): seal barely changes the lift (|ratio| small)."
              if abs(ratio) < 0.25 else
              "  SEAL-GATED: forcing seal=0 changes the lift substantially (thesis WRONG).")
    print(f"\nwrote {args.out/'lift_causality.json'}")


if __name__ == "__main__":
    main()
