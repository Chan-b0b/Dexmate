#!/usr/bin/env python3
"""Does the model gate descend-STOP on proprioceptive ee_z (a shortcut) instead of vision/contact?

Counterfactual on TRAINING frames, the mirror of probe_film_authority.py: hold the image + wrench
(contact) + everything else FIXED and OVERRIDE only ee_z (state idx 2), sweeping it. Then watch the
predicted dz.

  - SHORTCUT (ee_z-driven stop): dz ramps from strong descend (high ee_z) to ~0 (stop) around the
    habitual depth, and crosses ~0 at the SAME ee_z across base frames — i.e. the stop is decided by
    proprioceptive ee_z REGARDLESS of the (fixed) image that still shows the object below. This is the
    under-reach mechanism: on a deep object the arm hits its habitual ee_z and stops before contact.
  - VISION/contact-driven (healthy): dz is ~FLAT in ee_z — the fixed image still says "object below ->
    keep descending", so faking ee_z doesn't change the command.

Punchline = the CONTRAST with probe_film_authority: there d(dz)/d(c^) ~ 0 (model ignores contact);
if here d(dz)/d(ee_z) is LARGE, the descend channel is owned by the ee_z shortcut, not the condition.

Structural FiLM config MUST match the checkpoint (same envs as train/deploy):
  FILM_COND=contact,fz,seal FILM_MASK_FORCE=0 FILM_INJECT=suffix \
    /home/dexmate/vla_venv/bin/python probe_depth_sensitivity.py \
      --checkpoint outputs/film_v3_contactfzseal_nomask/checkpoints/last
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

VLA_DIR = Path(__file__).resolve().parent
EE_Z, F_LO, F_HI, DZ = 2, 9, 12, 2   # state idx: ee_z, force xyz; action idx: dz

import film_contact  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=VLA_DIR / "outputs/film_v3_contactfzseal_nomask/checkpoints/last")
    ap.add_argument("--dataset-root", type=Path, default=VLA_DIR / "datasets/lges_suction")
    ap.add_argument("--repo-id", default="local/lges_suction")
    ap.add_argument("--episode", type=int, default=None, help="(default: first case_pick episode)")
    ap.add_argument("--n-frames", type=int, default=3, help="base descent frames to sweep")
    ap.add_argument("--contact-n", type=float, default=14.5, help="|F| below this = pre-contact")
    ap.add_argument("--zlo", type=float, default=0.76)
    ap.add_argument("--zhi", type=float, default=1.06)
    ap.add_argument("--zstep", type=float, default=0.02)
    args = ap.parse_args()

    cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
    mask_force = os.environ.get("FILM_MASK_FORCE", "0") not in ("0", "false", "False")
    inject = os.environ.get("FILM_INJECT", "suffix")
    f0 = float(os.environ.get("FILM_F0", "12"))
    tau = float(os.environ.get("FILM_TAU", "10"))
    fz_tau = float(os.environ.get("FILM_FZ_TAU", "30"))
    print(f"[probe] FiLM cond={cond} inject={inject} mask_force={mask_force} F0={f0:.0f} "
          f"tau={tau:.0f} fz_tau={fz_tau:.0f}  ckpt={args.checkpoint}")

    wm, ws = film_contact.load_wrench_stats(args.dataset_root)
    sm, ss = film_contact.load_seal_stats(args.dataset_root)
    film_contact.apply("v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
                       contact_F0=f0, contact_tau=tau, fz_tau=fz_tau,
                       mask_force=mask_force, inject=inject)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    model_dir = args.checkpoint / "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.eval()
    policy.config.n_action_steps = 1
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    bounds = [(e["dataset_from_index"], e["dataset_to_index"]) for e in ds.meta.episodes]
    ep = args.episode
    if ep is None:
        for i, (lo, _) in enumerate(bounds):
            t = ds[lo]["task"].lower()
            if "case" in t and "pick" in t and "place" not in t:
                ep = i
                break
        ep = ep if ep is not None else 0
    lo, hi = bounds[ep]
    task = ds[lo]["task"]
    print(f"\nepisode {ep}: \"{task}\"  frames {lo}..{hi}")

    def predict_dz(frame, eez):
        # hold the image + wrench (contact) fixed; override ONLY ee_z. Fix the flow-matching
        # noise (seed) so Δdz across the sweep reflects ee_z alone, not sampling.
        policy.reset()
        torch.manual_seed(0)
        state = frame["observation.state"].clone().unsqueeze(0)
        if eez is not None:
            state[0, EE_Z] = eez
        obs = {"observation.state": state, "task": task}
        for k in frame:
            if k.startswith("observation.images."):
                obs[k] = frame[k].unsqueeze(0)
        with torch.inference_mode():
            a = policy.select_action(pre(obs))
        return float(post(a).squeeze(0).cpu().numpy()[DZ])

    # base frames: pre-contact descent frames (expert descending), evenly spaced
    desc = []
    for i in range(lo, hi):
        f = ds[i]
        st = f["observation.state"].numpy()
        if float(np.linalg.norm(st[F_LO:F_HI])) < args.contact_n and f["action"].numpy()[DZ] < -0.001:
            desc.append(i)
    if not desc:
        print("no pre-contact descent frames found.")
        return
    picks = [desc[round(k * (len(desc) - 1) / max(args.n_frames - 1, 1))]
             for k in range(args.n_frames)]
    picks = sorted(set(picks))
    frames = [ds[i] for i in picks]
    true_z = [float(f["observation.state"].numpy()[EE_Z]) for f in frames]
    print(f"base frames (idx @ true ee_z): " +
          "  ".join(f"{i}@{z:.3f}" for i, z in zip(picks, true_z)))
    print("each column = ONE fixed frame (image+contact frozen); rows sweep the FAKED ee_z.\n")

    zs = np.arange(args.zhi, args.zlo - 1e-9, -args.zstep)
    header = "  ee_z |" + "".join(f"  f{i}:dz(mm)" for i in picks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    grid = np.zeros((len(zs), len(frames)))
    for r, z in enumerate(zs):
        cells = []
        for c, fr in enumerate(frames):
            dz = predict_dz(fr, float(z)) * 1000
            grid[r, c] = dz
            cells.append(f"{dz:>9.1f}")
        mark = "  <- ~habitual stop" if abs(z - 0.82) < args.zstep / 2 else ""
        print(f"  {z:.3f} |" + "".join(cells) + mark)

    # per-frame: slope d(dz)/d(ee_z), dz range over the sweep, and the stop-threshold ee_z (dz>-1mm)
    print("\n  per base frame:")
    ranges = []
    for c, (i, z0) in enumerate(zip(picks, true_z)):
        col = grid[:, c]
        rng = col.max() - col.min()
        ranges.append(rng)
        dz_hi = col[int(np.argmin(np.abs(zs - 1.00)))]        # mid-descent (peak) vs habitual band
        dz_hab = col[int(np.argmin(np.abs(zs - 0.82)))]
        above = zs[col > -1.0]
        thr = above.min() if len(above) else float("nan")     # lowest faked ee_z where dz has ~stopped
        print(f"    f{i} (true ee_z {z0:.3f}): dz {col.min():.1f}..{col.max():.1f}mm "
              f"(range {rng:.1f}mm); dz@1.00={dz_hi:.1f} -> dz@0.82={dz_hab:.1f}mm; stops(dz>-1) by ee_z~{thr:.3f}")

    print("\n── verdict ─────────────────────────────────────────────────────────")
    mean_rng = float(np.mean(ranges))
    print(f"  faking ee_z alone moves dz by {mean_rng:.1f} mm (mean range over the sweep).")
    print(f"  compare probe_film_authority: forcing c^ 0->1 moved dz ~0.2 mm.")
    if mean_rng > 3.0:
        print("  => SHORTCUT CONFIRMED: the descend command is owned by proprioceptive ee_z, not the\n"
              "     (fixed) image/contact. On a deep object the arm stops at its habitual ee_z = before\n"
              "     contact. Conditioning can't fix this; need to break the ee_z cue / distill keep-descend.")
    else:
        print("  => ee_z does NOT dominate dz here — the model is using image/contact; look elsewhere.")


if __name__ == "__main__":
    main()
