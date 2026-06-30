#!/usr/bin/env python3
"""P1 — per-state-dimension action sensitivity (characterize contact-blindness).

For each of the 15 state dims, how much does the policy's action move when that
ONE feature is swung across its natural range, holding image + all other state
fixed (a single-feature do-intervention)? Binary dims (suction, vacuum_sealed)
are swung 0->1; continuous dims by +-1 std (from the take data). Broken down by
phase (approach / descent / post-seal lift) so we can see whether contact
features (suction, seal, wrench) are ignored everywhere or only somewhere.

Expectation from P0b: pose dims dominate; suction/seal/wrench are near-zero =
contact-blind. This probe makes that quantitative and per-phase, and adds the
wrench dims (untested before).

CAVEAT (learned 2026-06-17): swinging a feature to its range at an arbitrary
frame can create an OUT-OF-DISTRIBUTION input (e.g. seal=1 while the arm is
hovering at z=1.0, which never happens in data) and INFLATE the apparent
sensitivity. The aggregate `vacuum_sealed` number here (~122% of a step) is such
an artifact: a direct per-frame test shows seal-flip moves the action ~0.1mm at
the ACTUAL seal point (z=0.834 contact) but +6..12mm mid-descent (OOD). Read
sensitivity at in-distribution operating points, not the global swing.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/contact_aware_vla/state_sensitivity.py \
      [--max-takes 4] [--per-phase 5]
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
sys.path.insert(0, str(HERE.parent / "reactive_chunking"))  # reuse P0 loaders
from convert_to_lerobot import load_take, STATE_NAMES  # noqa: E402
from run_policy import load_policy, _to_chw  # noqa: E402
from p0_chunk_gap import _load_rgb, _load_depth  # noqa: E402

BINARY = {7, 8}  # suction, vacuum_sealed
CONTACT_DIMS = [7, 8, 9, 10, 11, 12, 13, 14]  # suction, seal, wrench fx..tz


def first_action(policy, pre, post, state, rgb_chw, depth_chw, instr, seed):
    obs = {"observation.images.head": rgb_chw,
           "observation.state": torch.from_numpy(state.astype(np.float32)).unsqueeze(0),
           "task": instr}
    if depth_chw is not None:
        obs["observation.images.head_depth"] = depth_chw
    obs = pre(obs)
    policy.reset()
    torch.manual_seed(seed)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(obs)
    return post(chunk[0][0:1]).squeeze(0).cpu().numpy()


def phase_indices(states, k):
    suc, seal = states[:, 7] > 0.5, states[:, 8] > 0.5
    n = len(states)
    son = int(np.argmax(suc)) if suc.any() else n
    sel = int(np.argmax(seal)) if seal.any() else n
    out = {}
    for name, (a, b) in {"approach": (0, son), "descend": (son, sel), "lift": (sel, n)}.items():
        if b > a:
            out[name] = np.unique(np.linspace(a, b - 1, min(k, b - a)).astype(int)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="case_pick")
    ap.add_argument("--recordings", type=Path, default=VLA_TRAIN.parent / "recordings")
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_TRAIN / "outputs/smolvla_depthseal_fixed/checkpoints/last")
    ap.add_argument("--max-takes", type=int, default=4)
    ap.add_argument("--per-phase", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    takes = sorted(p for p in (args.recordings / args.task).iterdir() if p.is_dir())[:args.max_takes]
    loaded = [(td, load_take(td, with_depth=True)) for td in takes]
    loaded = [(td, L) for td, L in loaded if L is not None]

    # per-dim mean/std over all frames (for continuous interventions) + action scale
    allstates = np.concatenate([L[3] for _, L in loaded])
    mean, std = allstates.mean(0), allstates.std(0)
    nat_step = float(np.mean([np.linalg.norm(L[4][:, :3], axis=1).mean() for _, L in loaded]))
    print(f"natural per-step dpos = {nat_step*1000:.2f} mm | "
          f"constant dims (std~0): {[STATE_NAMES[d] for d in range(15) if std[d] < 1e-6]}\n")

    policy, pre, post = load_policy(args.checkpoint)

    # sens[dim][phase] = list of ||dpos|| (m) induced by swinging that dim
    sens = {d: {} for d in range(15)}
    for ti, (td, L) in enumerate(loaded):
        instruction, rgb_paths, depth_paths, states, _ = L
        phs = phase_indices(states, args.per_phase)
        print(f"  [{ti+1}/{len(loaded)}] {td.name}: phases " +
              ", ".join(f"{k}={len(v)}f" for k, v in phs.items()))
        for phase, idxs in phs.items():
            for i in idxs:
                rgb = _to_chw(_load_rgb(rgb_paths[i]))
                depth = _to_chw(_load_depth(depth_paths[i]))
                for d in range(15):
                    if std[d] < 1e-6 and d not in BINARY:
                        continue
                    lo, hi = (0.0, 1.0) if d in BINARY else (mean[d] - std[d], mean[d] + std[d])
                    s_lo, s_hi = states[i].copy(), states[i].copy()
                    s_lo[d], s_hi[d] = lo, hi
                    a_lo = first_action(policy, pre, post, s_lo, rgb, depth, instruction, args.seed)
                    a_hi = first_action(policy, pre, post, s_hi, rgb, depth, instruction, args.seed)
                    sens[d].setdefault(phase, []).append(float(np.linalg.norm(a_hi[:3] - a_lo[:3])))

    # aggregate
    overall = {STATE_NAMES[d]: float(np.mean(np.concatenate([np.array(v) for v in sens[d].values()])) * 1000)
               if sens[d] else 0.0 for d in range(15)}
    by_phase_contact = {
        STATE_NAMES[d]: {ph: float(np.mean(v) * 1000) for ph, v in sens[d].items()}
        for d in CONTACT_DIMS}
    summary = {
        "natural_per_step_dpos_mm": nat_step * 1000,
        "overall_sensitivity_mm": overall,
        "overall_sensitivity_pct_of_step": {k: v / (nat_step * 1000) for k, v in overall.items()},
        "contact_dims_by_phase_mm": by_phase_contact,
    }
    (args.out / "state_sensitivity.json").write_text(json.dumps(summary, indent=2))
    print("\n===== P1 STATE SENSITIVITY (action dpos mm per natural feature swing) =====")
    for d in range(15):
        nm = STATE_NAMES[d]
        tag = "  <- contact" if d in CONTACT_DIMS else ""
        print(f"  {nm:14s} {overall[nm]:6.3f} mm  ({overall[nm]/(nat_step*1000)*100:5.1f}% of step){tag}")
    print("\n  -- contact dims by phase (mm) --")
    phases = ["approach", "descend", "lift"]
    print(f"  {'dim':14s} " + "".join(f"{p:>10s}" for p in phases))
    for d in CONTACT_DIMS:
        nm = STATE_NAMES[d]
        row = by_phase_contact[nm]
        print(f"  {nm:14s} " + "".join(f"{row.get(p, float('nan')):10.3f}" for p in phases))
    print(f"\nwrote {args.out/'state_sensitivity.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [STATE_NAMES[d] for d in range(15)]
        vals = [overall[n] for n in names]
        colors = ["tab:red" if d in CONTACT_DIMS else "tab:blue" for d in range(15)]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(names, vals, color=colors)
        ax.set(ylabel="action dpos sensitivity (mm)",
               title="per-state-dim action sensitivity (red = contact features)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(args.out / "state_sensitivity.png", dpi=120); plt.close(fig)
        print(f"wrote {args.out/'state_sensitivity.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
