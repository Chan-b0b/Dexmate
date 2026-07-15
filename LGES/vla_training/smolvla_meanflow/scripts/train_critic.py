#!/usr/bin/env python
# Copyright 2026. Licensed under the Apache License, Version 2.0.
"""MC-pretrain the value critic from ICM rewards on demo (and later rollout) data.

Cold-start step 1: the reversed-curiosity reward needs no environment, so demo
transitions are labeled offline with the frozen ICM and V(s) is regressed on
absorbing-tail normalized returns (see smolvla_meanflow/critic.py).

Accepts multiple feature caches (--features demo.npz rollouts.npz ...) so the
same script retrains the critic once phase-0 BC rollouts exist — that is when
it first sees informative negatives; on demos alone V is near-constant (~1) and
that is expected, not a bug.

  python train_critic.py --features icm_features_0708.npz \
      --icm ~/checkpoints/icm_0708_proprio.pt --out ~/checkpoints/critic_0708_proprio.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smolvla_meanflow.critic import CriticEnsemble  # noqa: E402
from smolvla_meanflow.icm import ICMEnsemble  # noqa: E402
from train_icm import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, nargs="+", required=True, help=".npz caches (demos [+ rollouts])")
    ap.add_argument("--icm", type=Path, required=True, help="frozen ICM checkpoint (rewards + feature variant)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--holdout-episodes", type=int, default=8, help="per cache, for val/early-stop")
    ap.add_argument("--ensemble-size", type=int, default=3)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    icm = ICMEnsemble.load(args.icm, map_location=device)
    icm.to(device)
    print(f"ICM: {args.icm} (variant={icm.variant}, eta={float(icm.eta):.4f})")

    # label every episode of every cache with rewards -> normalized returns
    feats_all, returns_all, val_mask_all = [], [], []
    for cache_path in args.features:
        cache = dict(np.load(cache_path))
        feats = build_features(cache, icm.variant)
        acts, ep_idx = cache["action"], cache["episode_index"]
        episodes = np.unique(ep_idx)
        val_eps = set(episodes[np.linspace(0, len(episodes) - 1, args.holdout_episodes).astype(int)])
        F = torch.as_tensor(feats, device=device)
        A = torch.as_tensor(acts, device=device)
        n_val = 0
        for ep in episodes:
            idx = np.nonzero(ep_idx == ep)[0]
            if len(idx) < 2:
                continue
            j = torch.as_tensor(idx, device=device)
            r = icm.reward(F[j[:-1]], A[j[:-1]], F[j[1:]])  # [T-1]
            v = CriticEnsemble.normalized_returns(r, args.gamma)
            feats_all.append(feats[idx[:-1]])
            returns_all.append(v.cpu().numpy())
            is_val = ep in val_eps
            val_mask_all.append(np.full(len(idx) - 1, is_val))
            n_val += is_val
        print(f"labeled {cache_path.name}: {len(episodes)} eps ({n_val} val)")

    X = np.concatenate(feats_all)
    Y = np.concatenate(returns_all).astype(np.float32)
    val = np.concatenate(val_mask_all)
    tr_i, va_i = np.nonzero(~val)[0], np.nonzero(val)[0]
    print(f"samples: train={len(tr_i)} val={len(va_i)}  returns: mean={Y.mean():.4f} "
          f"min={Y.min():.4f} p05={np.percentile(Y, 5):.4f}")

    critic = CriticEnsemble(
        feat_dim=X.shape[1], variant=icm.variant, ensemble_size=args.ensemble_size,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, gamma=args.gamma,
    ).to(device)
    critic.set_normalization(X[tr_i].mean(0), X[tr_i].std(0))

    Xt = torch.as_tensor(X, device=device)
    Yt = torch.as_tensor(Y, device=device)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    @torch.no_grad()
    def val_mse() -> float:
        errs = []
        for lo in range(0, len(va_i), 8192):
            j = torch.as_tensor(va_i[lo:lo + 8192], device=device)
            errs.append((critic.value(Xt[j]) - Yt[j]) ** 2)
        return float(torch.cat(errs).mean())

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(args.epochs):
        perms = [np.random.permutation(tr_i) for _ in critic.members]
        total, nb = 0.0, 0
        for lo in range(0, len(tr_i), args.batch_size):
            loss = 0.0
            for m, perm in zip(critic.members, perms):
                j = torch.as_tensor(perm[lo:lo + args.batch_size], device=device)
                nfeat = (Xt[j] - critic.feat_mean) / critic.feat_std
                loss = loss + ((m(nfeat) - Yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) / len(critic.members)
            nb += 1
        sched.step()
        if (epoch + 1) % 10 == 0:
            v = val_mse()
            if v < best_val:
                best_val, best_epoch = v, epoch + 1
                best_state = {k: t.detach().clone() for k, t in critic.state_dict().items()}
            if (epoch + 1) % max(10, args.epochs // 10) == 0:
                print(f"epoch {epoch + 1:4d}  train_mse={total / max(nb, 1):.6f}  "
                      f"val_mse={v:.6f}  (best {best_val:.6f} @ {best_epoch})")

    if best_state is not None:
        critic.load_state_dict(best_state)
        print(f"restored best snapshot: epoch {best_epoch} (val_mse={best_val:.6f})")
    critic.eval()

    # advantage preview on val: A_t = target - V(s_t); also positive value loss (regret proxy)
    with torch.no_grad():
        j = torch.as_tensor(va_i, device=device)
        adv = (Yt[j] - critic.value(Xt[j])).cpu().numpy()
    pvl = np.maximum(adv, 0)
    print(f"val advantages: mean={adv.mean():+.4f} std={adv.std():.4f} "
          f"| positive value loss mean={pvl.mean():.4f} p95={np.percentile(pvl, 95):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    critic.save(args.out, extra={
        "features_caches": [str(p) for p in args.features], "icm": str(args.icm),
        "variant": icm.variant, "gamma": args.gamma, "best_epoch": best_epoch,
        "val_mse": best_val, "returns_mean": float(Y.mean()), "returns_p05": float(np.percentile(Y, 5)),
        "seed": args.seed, "epochs": args.epochs,
    })
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
