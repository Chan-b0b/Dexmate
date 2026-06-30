#!/usr/bin/env python3
"""Train the LGES latent dynamics model on the existing LeRobot dataset.

Milestone 1: offline world model only (no reward/value/planner). Success =
multi-step prediction of wrench + seal that beats the persistence baseline
(see validate.py).

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python MPC/train.py
  /home/dexmate/vla_venv/bin/python MPC/train.py --epochs 80 --horizon 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_episodes, Normalizer, WindowDataset  # noqa: E402
from model import LatentDynamics, rollout_loss              # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO / "LGES/vla_training/datasets/lges_suction"
DEFAULT_VAL = REPO / "LGES/vla_training/datasets/lges_suction_val"


def run_epoch(model, loader, device, opt=None, **loss_kw):
    train = opt is not None
    model.train(train)
    totals, n = {}, 0
    for states, actions in loader:
        states, actions = states.to(device), actions.to(device)
        with torch.set_grad_enabled(train):
            loss, agg = rollout_loss(model, states, actions, **loss_kw)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        bs = states.shape[0]
        n += bs
        for k, v in agg.items():
            totals[k] = totals.get(k, 0.0) + v * bs
    return {k: v / n for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs/dyn")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--consistency", type=float, default=10.0)
    ap.add_argument("--seal-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading episodes from {args.data}")
    train_eps = load_episodes(args.data)
    val_eps = load_episodes(args.val)
    norm = Normalizer.fit(train_eps)
    train_ds = WindowDataset(train_eps, args.horizon, norm)
    val_ds = WindowDataset(val_eps, args.horizon, norm)
    print(f"train: {len(train_eps)} eps -> {len(train_ds)} windows | "
          f"val: {len(val_eps)} eps -> {len(val_ds)} windows | device {device}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = LatentDynamics(args.latent, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    loss_kw = dict(w_consistency=args.consistency, seal_weight=args.seal_weight)
    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        tr = run_epoch(model, train_dl, device, opt, **loss_kw)
        va = run_epoch(model, val_dl, device, None, **loss_kw)
        sched.step()
        log.append({"epoch": ep, "train": tr, "val": va})
        msg = (f"ep {ep:3d} | train {tr['total']:.4f} val {va['total']:.4f} "
               f"| val wrench {va['wrench']:.4f} sealed {va['sealed']:.4f} "
               f"cons {va['consistency']:.4f} | {time.time()-t0:.0f}s")
        if va["total"] < best:
            best = va["total"]
            torch.save({"model": model.state_dict(), "norm": norm.to_dict(),
                        "config": {"latent": args.latent, "hidden": args.hidden,
                                   "horizon": args.horizon}},
                       args.out / "best.pt")
            msg += "  *"
        print(msg)

    (args.out / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"done. best val total {best:.4f} -> {args.out/'best.pt'}")


if __name__ == "__main__":
    main()
