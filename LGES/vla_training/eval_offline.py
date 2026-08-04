#!/usr/bin/env python3
"""Open-loop offline eval of a trained SmolVLA checkpoint on held-out takes.

For every val episode: reset the policy, feed observations frame-by-frame
exactly as deployment would (same preprocessor/postprocessor as training),
call select_action, and compare the predicted action to the recorded one.

This is the cheap, no-robot sanity check: it answers "did the policy learn
the mapping?" but NOT "does it work closed-loop?" (errors compound on-robot;
only a real run shows that). Actions are next-state EE deltas, so errors are
reported in physical units per 1/15 s step.

  /home/dexmate/vla_venv/bin/python eval_offline.py \
      [--checkpoint outputs/<run>/checkpoints/last] \
      [--val-root datasets/lges_suction_val]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

VLA_DIR = Path(__file__).resolve().parent


def latest_checkpoint() -> Path:
    runs = sorted((VLA_DIR / "outputs").glob("*/checkpoints/last"))
    if not runs:
        raise SystemExit("no checkpoints under outputs/*/checkpoints/last")
    return runs[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="checkpoint dir (default: latest outputs/*/checkpoints/last)")
    ap.add_argument("--val-root", type=Path, default=VLA_DIR / "datasets" / "lges_suction_val")
    ap.add_argument("--repo-id", default="local/lges_suction_val")
    ap.add_argument("--film", action="store_true",
                    help="evaluate a FiLM checkpoint (apply film_contact before load; "
                         "FILM_COND/F0/TAU/FZ_TAU/MASK_FORCE/INJECT envs MUST match training)")
    ap.add_argument("--stats-root", type=Path, default=None,
                    help="dataset whose stats feed c-hat (default: --val-root); set to the "
                         "TRAINING dataset to match training exactly")
    args = ap.parse_args()

    ckpt = args.checkpoint or latest_checkpoint()
    model_dir = ckpt / "pretrained_model"
    print(f"checkpoint: {ckpt}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    if args.film:
        import os
        import film_contact
        mask_force = os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False")
        cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
        inject = os.environ.get("FILM_INJECT", "suffix")
        f0 = float(os.environ.get("FILM_F0", "12"))
        tau = float(os.environ.get("FILM_TAU", "10"))
        fz_tau = float(os.environ.get("FILM_FZ_TAU", "30"))
        fz_off = float(os.environ.get("FILM_FZ_OFF", "2.6"))
        fmag_off = float(os.environ.get("FILM_FMAG_OFF", "5.1"))
        fmag_tau = float(os.environ.get("FILM_FMAG_TAU", "5"))
        stats_root = args.stats_root or args.val_root
        wm, ws = film_contact.load_wrench_stats(stats_root)
        sm, ss = film_contact.load_seal_stats(stats_root)
        dm, dsd = film_contact.load_dfmag_stats(stats_root)
        film_contact.apply("v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
                           contact_F0=f0, contact_tau=tau, fz_tau=fz_tau, fz_off=fz_off,
                           mask_force=mask_force, inject=inject,
                           dfmag_mean=dm, dfmag_std=dsd,
                           dfmag_tau=float(os.environ.get("FILM_DFMAG_TAU", "5")),
                           fmag_off=fmag_off, fmag_tau=fmag_tau)
        print(f"[eval] FiLM ENABLED (cond={cond} inject={inject} mask_force={mask_force} "
              f"F0={f0:.0f} tau={tau:.0f} fz_tau={fz_tau:.0f} fz_off={fz_off:g} "
              f"fmag={fmag_off:g}/{fmag_tau:g} stats={stats_root})")

    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.eval()
    device = policy.config.device
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    ds = LeRobotDataset(args.repo_id, root=args.val_root)
    bounds = [(ep["dataset_from_index"], ep["dataset_to_index"]) for ep in ds.meta.episodes]
    print(f"val: {ds.num_episodes} episodes, {ds.num_frames} frames\n")

    # per-task accumulation of |predicted - recorded| action errors
    per_task = defaultdict(list)

    for ep, (lo, hi) in enumerate(bounds):
        policy.reset()
        task = ds[lo]["task"]
        errs = []
        for i in range(lo, hi):
            frame = ds[i]
            obs = {"observation.state": frame["observation.state"].unsqueeze(0), "task": task}
            for k in frame:                    # all cameras (0729 sets use camera1/2/3)
                if k.startswith("observation.images."):
                    obs[k] = frame[k].unsqueeze(0)
            obs = pre(obs)
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = post(action).squeeze(0).cpu().numpy()
            errs.append(np.abs(action - frame["action"].numpy()))
        per_task[task].append(np.array(errs))

    # ── report ────────────────────────────────────────────────────────
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "suction"]
    all_errs = []
    print(f"{'task':<42} {'pos(mm)':>8} {'rot(mrad)':>10} {'suction':>8}")
    for task, eps in sorted(per_task.items()):
        e = np.concatenate(eps)  # [N,7]
        all_errs.append(e)
        pos_mm = e[:, :3].mean() * 1000
        rot_mr = e[:, 3:6].mean() * 1000
        suc_acc = (e[:, 6] < 0.5).mean() * 100  # rounded suction matches
        print(f"{task:<42} {pos_mm:>8.2f} {rot_mr:>10.2f} {suc_acc:>7.1f}%")

    e = np.concatenate(all_errs)
    print("-" * 72)
    print(f"{'OVERALL mean |err| per step':<42} {e[:, :3].mean()*1000:>8.2f} "
          f"{e[:, 3:6].mean()*1000:>10.2f} {(e[:, 6] < 0.5).mean()*100:>7.1f}%")
    print("\nper-dimension mean |err|:")
    for n, v in zip(names, e.mean(0)):
        print(f"  {n:<8} {v:.5f}")


if __name__ == "__main__":
    main()
