#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""Train a reversed-curiosity ICM (forward-model ensemble) from a feature cache.

Consumes the .npz written by extract_icm_features.py, builds per-step
transitions (s_t, a_t, s_t+1) within episode boundaries, trains an ensemble per
feature variant, and calibrates the reward scale eta on HELD-OUT demo episodes
so that in-distribution states score near r = 1.

  python train_icm.py --features icm_features_0708.npz --variant proprio \
      --out ~/checkpoints/icm_0708_proprio.pt

Calibration: eta is set so the held-out p95 error maps to r = --r-at-p95
(default 0.8). Held-out demo frames are in-distribution by construction, so
rewards on them should sit near 1; states the ICM has never seen score lower.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smolvla_meanflow.icm import ICMEnsemble, VARIANTS  # noqa: E402


def build_features(cache: dict, variant: str) -> np.ndarray:
    if variant == "proprio":
        return cache["state"]
    if "vision" not in cache:
        raise ValueError(f"variant {variant!r} needs vision features; re-run extract without --no-vision")
    vision = cache["vision"].astype(np.float32)
    if variant == "vision":
        return vision
    return np.concatenate([cache["state"], vision], axis=-1)  # both


def transitions(episode_index: np.ndarray) -> np.ndarray:
    """Indices t where t and t+1 belong to the same episode."""
    same = episode_index[:-1] == episode_index[1:]
    return np.nonzero(same)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True, help=".npz from extract_icm_features.py")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-episodes", type=int, default=8)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--r-at-p95", type=float, default=0.8, help="reward at held-out p95 error (sets eta)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = dict(np.load(args.features))
    feats = build_features(cache, args.variant)
    acts = cache["action"]
    ep_idx = cache["episode_index"]
    t_idx = transitions(ep_idx)

    # episode-wise split: held-out episodes measure the in-distribution noise floor.
    # Half the held-out episodes drive early stopping (val), the other half stay
    # untouched for eta calibration (cal) so the reward scale isn't fit to the
    # same data that picked the snapshot.
    episodes = np.unique(ep_idx)
    hold_eps = episodes[np.linspace(0, len(episodes) - 1, args.holdout_episodes).astype(int)]
    val_eps, cal_eps = hold_eps[::2], hold_eps[1::2]
    tr = t_idx[~np.isin(ep_idx[t_idx], hold_eps)]
    va = t_idx[np.isin(ep_idx[t_idx], val_eps)]
    ca = t_idx[np.isin(ep_idx[t_idx], cal_eps)]
    print(f"variant={args.variant}  feat_dim={feats.shape[1]}  transitions: train={len(tr)} "
          f"val={len(va)} (eps {val_eps.tolist()})  cal={len(ca)} (eps {cal_eps.tolist()})  device={device}")

    icm = ICMEnsemble(
        feat_dim=feats.shape[1], action_dim=acts.shape[1], variant=args.variant,
        ensemble_size=args.ensemble_size, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    ).to(device)
    icm.set_normalization(feats[tr].mean(0), feats[tr].std(0), acts[tr].mean(0), acts[tr].std(0))

    F = torch.as_tensor(feats, device=device)
    A = torch.as_tensor(acts, device=device)
    opt = torch.optim.AdamW(icm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def errors_on(idx: np.ndarray) -> torch.Tensor:
        errs = []
        for lo in range(0, len(idx), 4096):
            j = torch.as_tensor(idx[lo:lo + 4096], device=device)
            errs.append(icm.prediction_error(F[j], A[j], F[j + 1]))
        return torch.cat(errs)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(args.epochs):
        # each member sees its own shuffle — cheap decorrelation for the ensemble
        perms = [np.random.permutation(tr) for _ in icm.members]
        total = 0.0
        for lo in range(0, len(tr), args.batch_size):
            loss = 0.0
            for m, perm in zip(icm.members, perms):
                j = torch.as_tensor(perm[lo:lo + args.batch_size], device=device)
                nf, na = icm.normalize(F[j], A[j])
                pred = m(nf, na)
                loss = loss + ((pred - icm.target_delta(F[j], F[j + 1])) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) / len(icm.members)
        sched.step()
        if (epoch + 1) % 10 == 0:
            ve = errors_on(va)
            val = float(ve.mean())
            if val < best_val:
                best_val, best_epoch = val, epoch + 1
                best_state = {k: v.detach().clone() for k, v in icm.state_dict().items()}
            if (epoch + 1) % max(10, args.epochs // 10) == 0:
                print(f"epoch {epoch + 1:4d}  train_mse={total / max(1, len(tr) // args.batch_size):.5f}  "
                      f"val mean={val:.5f} p95={ve.quantile(0.95):.5f}  (best {best_val:.5f} @ {best_epoch})")

    if best_state is not None:
        icm.load_state_dict(best_state)
        print(f"restored best snapshot: epoch {best_epoch} (val mean={best_val:.5f})")
    icm.eval()
    he = errors_on(ca).cpu().numpy()
    pct = {p: float(np.percentile(he, p)) for p in (50, 90, 95, 99)}
    eta = -np.log(args.r_at_p95) / max(pct[95], 1e-12)
    icm.eta.fill_(eta)
    r_pct = {p: float(np.exp(-eta * v)) for p, v in pct.items()}
    print(f"calibration error percentiles: {pct}")
    print(f"eta={eta:.4f}  (r at p50/p90/p95/p99 = "
          f"{r_pct[50]:.3f}/{r_pct[90]:.3f}/{r_pct[95]:.3f}/{r_pct[99]:.3f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    icm.save(args.out, extra={
        "features_cache": str(args.features), "variant": args.variant,
        "state_dim": int(cache["state"].shape[1]),
        "vision_dim": int(cache["vision"].shape[1]) if "vision" in cache else 0,
        "val_episodes": val_eps.tolist(), "cal_episodes": cal_eps.tolist(),
        "cal_error_percentiles": pct, "eta": float(eta), "r_at_p95": args.r_at_p95,
        "best_epoch": best_epoch, "seed": args.seed, "epochs": args.epochs,
    })
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
