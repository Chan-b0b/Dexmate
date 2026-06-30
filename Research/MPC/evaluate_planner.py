#!/usr/bin/env python3
"""Offline verification of the MPC planner (no robot).

Two checks, on the held-out episodes:

  1. ACTION AGREEMENT — at each true recorded state, plan and compare the
     planned motion to what the demo actually did (direction cosine + suction
     match), split pre/post seal. A sanity check that the planner moves the way
     a successful demo moves, not that it is identical.

  2. IMAGINED CLOSED-LOOP — from a recorded start state, let the planner control
     the MODEL'S OWN predicted dynamics for N steps and check the behavior
     composes: pick = descend -> model predicts seal -> lift; place = descend
     while holding -> release -> lift. This tests reward+planner+model together.
     It does NOT prove real-world success (that needs the robot and guards
     against model exploitation).

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python MPC/evaluate_planner.py
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Normalizer, POS, QUAT, SUCTION, SEALED, WRENCH      # noqa: E402
from model import LatentDynamics                                     # noqa: E402
from reward import compute_targets, PhaseReward, PHASE_KIND          # noqa: E402
from planner import MPPIPlanner                                      # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO / "LGES/vla_training/datasets/lges_suction"
DEFAULT_VAL = REPO / "LGES/vla_training/datasets/lges_suction_val"
_FZ = WRENCH.start + 2


def load_with_phase(root):
    cols = ["observation.state", "action", "episode_index", "frame_index", "task_index"]
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    phases = list(PHASE_KIND)
    out = []
    for _, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        s = np.stack(g["observation.state"].values).astype(np.float32)
        a = np.stack(g["action"].values).astype(np.float32)
        out.append((s, a, phases[int(g["task_index"].iloc[0])]))
    return out


def quat_step(q_wxyz, drot):
    q = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    qn = (Rotation.from_rotvec(drot) * q).as_quat()  # xyzw
    return np.array([qn[3], qn[0], qn[1], qn[2]], np.float32)


@torch.no_grad()
def imagine(planner, model, norm, start, steps, device):
    """Planner controls the model's predicted dynamics. Returns per-step
    (ee_z, sealed_prob, fz)."""
    planner.reset()
    s = start.copy()
    s_mean = norm.s_mean
    traj = []
    for _ in range(steps):
        a = planner.plan(s)
        zn = torch.tensor(norm.norm_state(s), device=device)[None]
        z = model.step(model.encode(zn), torch.tensor(norm.norm_action(a), device=device)[None])
        dec = model.decode(z)[0]
        sealed_p = torch.sigmoid(dec[SEALED]).item()
        dec_d = norm.denorm_state(dec.cpu().numpy())
        nxt = dec_d.copy()
        nxt[POS] = s[POS] + a[0:3]                      # analytic pose
        nxt[QUAT] = quat_step(s[QUAT], a[3:6])
        nxt[SUCTION] = a[6]
        nxt[SEALED] = float(sealed_p > 0.5)
        traj.append((nxt[2], sealed_p, dec_d[_FZ]))
        s = nxt
    return np.array(traj)


def action_agreement(planner, episodes, stride=4):
    """Direction cosine + suction match between planned and demo actions."""
    pre, post = {"cos": [], "suc": []}, {"cos": [], "suc": []}
    for s, a, _ in episodes:
        planner.reset()
        for t in range(0, len(a) - 1, stride):
            ap = planner.plan(s[t])
            dp, dd = ap[0:3], a[t, 0:3]
            nd = np.linalg.norm(dp) * np.linalg.norm(dd)
            cos = float(dp @ dd / nd) if nd > 1e-6 else np.nan
            suc = float((ap[6] >= 0.5) == (a[t, 6] >= 0.5))
            bucket = post if s[t, SEALED] > 0.5 else pre
            bucket["cos"].append(cos)
            bucket["suc"].append(suc)
    return pre, post


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path(__file__).resolve().parent / "runs/dyn/best.pt")
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="for data-derived targets")
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs/dyn")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--imagine-steps", type=int, default=160)
    ap.add_argument("--phases", nargs="+", default=None, help="subset, e.g. case_pick case_place")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = LatentDynamics(cfg["latent"], cfg["hidden"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    norm = Normalizer.from_dict(blob["norm"])
    targets = compute_targets(args.train)
    episodes = load_with_phase(args.val)
    print(f"loaded {args.ckpt} | {len(episodes)} val episodes | device {device}\n")

    phases = args.phases or list(PHASE_KIND)
    plt = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pass

    print(f"{'phase':16s} {'kind':5s} | {'pre-seal cos|suc':>18} {'post-seal cos|suc':>18} "
          f"| {'imagined: dz↓':>13} {'seal max':>9} {'lifted?':>8}")
    for phase in phases:
        reward = PhaseReward(targets[phase])
        planner = MPPIPlanner(model, norm, reward, device, args.horizon, args.samples)
        eps = [e for e in episodes if e[2] == phase]
        if not eps:
            continue
        pre, post = action_agreement(planner, eps)
        cp = np.nanmean(pre["cos"]) if pre["cos"] else np.nan
        sp = np.mean(pre["suc"]) if pre["suc"] else np.nan
        cq = np.nanmean(post["cos"]) if post["cos"] else np.nan
        sq = np.mean(post["suc"]) if post["suc"] else np.nan

        tr = imagine(planner, model, norm, eps[0][0][0], args.imagine_steps, device)
        z0, zmin = tr[0, 0], tr[:, 0].min()
        seal_max = tr[:, 1].max()
        seal_idx = np.argmax(tr[:, 1] > 0.5) if seal_max > 0.5 else None
        lifted = (tr[seal_idx:, 0].max() - zmin > 0.05) if seal_idx is not None else False
        print(f"{phase:16s} {reward.kind:5s} | {cp:>8.2f}|{sp:<8.2f} {cq:>8.2f}|{sq:<8.2f} "
              f"| {z0-zmin:>12.3f} {seal_max:>9.2f} {str(lifted):>8}")

        if plt is not None:
            t = np.arange(len(tr))
            fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            ax[0].plot(t, tr[:, 0]); ax[0].axhline(targets[phase]["contact"][2], ls=":", c="r", label="contact z")
            ax[0].set_ylabel("ee z (m)"); ax[0].legend(); ax[0].set_title(f"{phase} — imagined closed-loop (planner ⟶ model)")
            ax[1].plot(t, tr[:, 1], label="seal prob"); ax[1].plot(t, tr[:, 2] / 20.0, label="fz/20")
            ax[1].set_ylabel("seal / fz"); ax[1].set_xlabel("planner step (15 Hz)"); ax[1].legend()
            fig.tight_layout(); fig.savefig(args.out / f"plan_{phase}.png", dpi=110); plt.close(fig)

    if plt is not None:
        print(f"\nsaved imagined-rollout plots -> {args.out}/plan_*.png")


if __name__ == "__main__":
    main()
