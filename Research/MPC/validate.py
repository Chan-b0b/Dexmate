#!/usr/bin/env python3
"""Validate the LGES latent dynamics model: multi-step open-loop prediction.

Encodes a state, rolls the latent dynamics forward with the TRUE recorded
actions, decodes, and measures error vs. horizon on the held-out episodes —
broken out by group, because pos/quat/suction are analytic functions of the
action (should be ~exact) while wrench + the seal event are the only genuinely
learned dynamics. Everything is compared against a persistence baseline
(predict the start state); the model has to beat it on wrench + seal to be worth
planning with.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python MPC/validate.py
  /home/dexmate/vla_venv/bin/python MPC/validate.py --ckpt MPC/runs/dyn/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (load_episodes, Normalizer, POS, QUAT, SUCTION, SEALED, WRENCH)  # noqa: E402
from model import LatentDynamics                                                 # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VAL = REPO / "LGES/vla_training/datasets/lges_suction_val"
FPS = 15.0


def load_model(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = LatentDynamics(cfg["latent"], cfg["hidden"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, Normalizer.from_dict(blob["norm"])


@torch.no_grad()
def error_vs_horizon(model, norm, episodes, device, max_h, stride):
    """Aggregate per-offset prediction error over all start points / episodes."""
    keys = ["pos_mm", "quat_deg", "wrench_N", "fz_N", "seal_acc", "seal_brier",
            "suc_acc", "b_pos_mm", "b_wrench_N", "b_fz_N", "b_seal_acc", "cnt"]
    acc = {k: np.zeros(max_h) for k in keys}
    smean = torch.tensor(norm.s_mean, device=device)
    sstd = torch.tensor(norm.s_std, device=device)

    for s, a in episodes:
        K = len(s)
        if K < 3:
            continue
        S = torch.tensor(norm.norm_state(s), device=device)
        A = torch.tensor(norm.norm_action(a), device=device)
        starts = torch.arange(0, K - 1, stride, device=device)
        z = model.encode(S[starts])
        start_d = S[starts] * sstd + smean          # persistence prediction (denorm)

        for k in range(max_h):
            idx = starts + k
            valid = idx <= (K - 2)
            if not valid.any():
                break
            z = model.step(z, A[idx.clamp(max=K - 2)])
            pred = model.decode(z)
            tgt = S[(starts + k + 1).clamp(max=K - 1)]
            pred_d, tgt_d = pred * sstd + smean, tgt * sstd + smean

            pos = (pred_d[:, POS] - tgt_d[:, POS]).norm(dim=1) * 1000.0
            qp = torch.nn.functional.normalize(pred[:, QUAT], dim=1)
            quat = 2 * torch.arccos((qp * tgt[:, QUAT]).sum(1).abs().clamp(max=1.0)) * 180 / np.pi
            wr = (pred_d[:, WRENCH] - tgt_d[:, WRENCH]).abs()
            seal_p = torch.sigmoid(pred[:, SEALED])
            seal_t = tgt[:, SEALED]
            suc_ok = ((torch.sigmoid(pred[:, SUCTION]) > 0.5).float() == tgt[:, SUCTION]).float()
            b_wr = (start_d[:, WRENCH] - tgt_d[:, WRENCH]).abs()
            b_pos = (start_d[:, POS] - tgt_d[:, POS]).norm(dim=1) * 1000.0
            b_seal_ok = ((start_d[:, SEALED] > 0.5).float() == seal_t).float()

            m = valid
            def add(key, val):
                acc[key][k] += val[m].sum().item()
            add("pos_mm", pos); add("quat_deg", quat)
            add("wrench_N", wr.mean(1)); add("fz_N", wr[:, 2])
            add("seal_acc", ((seal_p > 0.5).float() == seal_t).float())
            add("seal_brier", (seal_p - seal_t) ** 2)
            add("suc_acc", suc_ok)
            add("b_pos_mm", b_pos); add("b_wrench_N", b_wr.mean(1)); add("b_fz_N", b_wr[:, 2])
            add("b_seal_acc", b_seal_ok)
            acc["cnt"][k] += m.sum().item()

    cnt = np.maximum(acc.pop("cnt"), 1)
    return {k: v / cnt for k, v in acc.items()}, cnt


@torch.no_grad()
def full_rollout(model, norm, s, a, device):
    """Open-loop rollout from t=0 over the whole episode. Returns predicted
    denormalized states and seal probabilities aligned to true indices 1..K-1."""
    smean = torch.tensor(norm.s_mean, device=device)
    sstd = torch.tensor(norm.s_std, device=device)
    S = torch.tensor(norm.norm_state(s), device=device)
    A = torch.tensor(norm.norm_action(a), device=device)
    z = model.encode(S[:1])
    preds = []
    for i in range(len(s) - 1):
        z = model.step(z, A[i:i + 1])
        preds.append(model.decode(z))
    P = torch.cat(preds, 0)
    seal_prob = torch.sigmoid(P[:, SEALED]).cpu().numpy()
    return (P * sstd + smean).cpu().numpy(), seal_prob


def first_toggle(series):
    """Index of the first 0/1 change in a binary series, or None."""
    d = np.where(series[1:] != series[:-1])[0]
    return int(d[0] + 1) if len(d) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path(__file__).resolve().parent / "runs/dyn/best.pt")
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs/dyn")
    ap.add_argument("--max-h", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--report-offsets", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--plot-episodes", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, norm = load_model(args.ckpt, device)
    episodes = load_episodes(args.val)
    print(f"loaded {args.ckpt} | {len(episodes)} val episodes | device {device}\n")

    curves, cnt = error_vs_horizon(model, norm, episodes, device, args.max_h, args.stride)

    offs = [o for o in args.report_offsets if o <= args.max_h and cnt[o - 1] > 0]
    print("Multi-step open-loop error vs horizon (model | persistence baseline):")
    print(f"{'k':>4} {'ms':>6} {'pos mm':>16} {'wrench N':>16} {'fz N':>16} "
          f"{'seal acc':>16} {'quat°':>7}")
    for o in offs:
        i = o - 1
        print(f"{o:>4} {o/FPS*1000:>6.0f} "
              f"{curves['pos_mm'][i]:>7.1f}|{curves['b_pos_mm'][i]:<8.1f} "
              f"{curves['wrench_N'][i]:>7.2f}|{curves['b_wrench_N'][i]:<8.2f} "
              f"{curves['fz_N'][i]:>7.2f}|{curves['b_fz_N'][i]:<8.2f} "
              f"{curves['seal_acc'][i]:>7.3f}|{curves['b_seal_acc'][i]:<8.3f} "
              f"{curves['quat_deg'][i]:>7.2f}")

    # full-episode rollout: seal-toggle timing + plots
    timing_err, toggles = [], 0
    for s, a in episodes:
        pred_d, seal_prob = full_rollout(model, norm, s, a, device)
        true_seal = s[1:, SEALED]                       # aligned to pred indices
        ta, tp = first_toggle(true_seal), first_toggle((seal_prob > 0.5).astype(np.float32))
        if ta is not None:
            toggles += 1
            if tp is not None:
                timing_err.append(abs(tp - ta))
    onset_ms = float(np.mean(timing_err)) / FPS * 1000 if timing_err else None

    print(f"\nSeal-toggle timing (full open-loop rollout): "
          f"{len(timing_err)}/{toggles} episodes detected, "
          f"mean |Δ| = {np.mean(timing_err):.1f} frames "
          f"({onset_ms:.0f} ms)" if timing_err else "\nno seal toggles detected")

    # save metrics
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = {"offsets": offs,
               "curves": {k: v.tolist() for k, v in curves.items()},
               "seal_toggle_mae_frames": float(np.mean(timing_err)) if timing_err else None,
               "seal_toggle_detected": [len(timing_err), toggles]}
    (args.out / "val_metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for n in range(min(args.plot_episodes, len(episodes))):
            s, a = episodes[n]
            pred_d, seal_prob = full_rollout(model, norm, s, a, device)
            t = np.arange(len(seal_prob))
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax[0].plot(t, s[1:, WRENCH.start + 2], label="fz true", lw=1.5)
            ax[0].plot(t, pred_d[:, WRENCH.start + 2], "--", label="fz pred")
            ax[0].set_ylabel("fz (N)"); ax[0].legend(); ax[0].set_title(f"val episode {n} — open-loop from t=0")
            ax[1].plot(t, s[1:, SEALED], label="sealed true", lw=1.5)
            ax[1].plot(t, seal_prob, "--", label="seal prob")
            ax[1].set_ylabel("sealed"); ax[1].set_xlabel("frame (15 Hz)"); ax[1].legend()
            fig.tight_layout(); fig.savefig(args.out / f"rollout_ep{n}.png", dpi=110)
            plt.close(fig)
        print(f"saved plots + val_metrics.json -> {args.out}")
    except ImportError:
        print(f"saved val_metrics.json -> {args.out} (matplotlib unavailable, skipped plots)")


if __name__ == "__main__":
    main()
