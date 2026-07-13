#!/usr/bin/env python
"""AWR learner: rescans the replay buffer, fits a critic, updates the residual policy by
advantage-weighted regression, and periodically checkpoints for actor.py to pick up.

Runs as its OWN process (separate from actor.py's 15 Hz control loop) so gradient steps
never compete with the robot tick budget -- see the residual_rl design discussion. The
actor only ever reads the latest checkpoint, and only between episodes.

  python learner.py --buffer-dir ~/rl_buffer/case_pick --ckpt-dir ~/rl_ckpt/case_pick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from replay_buffer import Episode, ReplayBuffer
from residual_policy import ResidualPolicy, save_atomic


class ValueFunction(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def episode_returns(ep: Episode, gamma: float, value_fn: ValueFunction, device: str) -> np.ndarray:
    """Backward discounted return-to-go. Terminal episodes end with reward-only (no
    bootstrap); truncated episodes (terminal=False) bootstrap the tail with the CURRENT
    critic at last_next_state -- the state one tick past the last logged transition, so
    the bootstrap isn't off by one (see replay_buffer.py's docstring on why that state is
    stored explicitly instead of reused from the last logged row)."""
    T = len(ep.rewards)
    returns = np.zeros(T, dtype=np.float32)
    if ep.terminal:
        bootstrap = 0.0
    else:
        assert ep.last_next_state is not None, "truncated episode missing last_next_state"
        with torch.no_grad():
            s = torch.as_tensor(ep.last_next_state, device=device).unsqueeze(0)
            bootstrap = float(value_fn(s).squeeze(0))
    running = bootstrap
    for t in range(T - 1, -1, -1):
        running = ep.rewards[t] + gamma * running
        returns[t] = running
    return returns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer-dir", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--obs-dim", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--beta", type=float, default=0.5, help="AWR temperature (advantage / beta)")
    ap.add_argument("--weight-clip", type=float, default=20.0, help="clip |advantage/beta| before exp")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--min-transitions", type=int, default=500,
                    help="don't start training until the buffer has at least this many steps")
    ap.add_argument("--steps-per-refresh", type=int, default=50,
                    help="gradient steps between buffer rescans")
    ap.add_argument("--refresh-interval-s", type=float, default=5.0,
                    help="min seconds between buffer rescans (caps rescan overhead)")
    ap.add_argument("--save-every", type=int, default=200, help="gradient steps between checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None, help="residual policy checkpoint to resume from")
    args = ap.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()

    policy = (ResidualPolicy.load(args.resume, map_location=args.device) if args.resume
              else ResidualPolicy(obs_dim=args.obs_dim)).to(args.device)
    value_fn = ValueFunction(args.obs_dim).to(args.device)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(value_fn.parameters(), lr=args.lr)

    buf = ReplayBuffer(args.buffer_dir)
    added = buf.refresh()
    print(f"[learner] initial scan: {len(buf.episodes)} closed episodes, {len(buf)} transitions")

    step = 0
    last_refresh = time.time()
    while True:
        if len(buf) < args.min_transitions:
            print(f"[learner] waiting for data: {len(buf)}/{args.min_transitions} transitions")
            time.sleep(args.refresh_interval_s)
            buf.refresh()
            continue

        for _ in range(args.steps_per_refresh):
            batch = buf.sample_transitions(args.batch_size, rng)
            if not batch:
                break
            states, res_actions, returns = [], [], []
            # group by episode so episode_returns (which needs a full backward pass) is
            # computed once per episode per batch, not once per sampled transition.
            by_ep: dict[str, list[int]] = {}
            for ep_id, t in batch:
                by_ep.setdefault(ep_id, []).append(t)
            for ep_id, ts in by_ep.items():
                ep = buf.episodes[ep_id]
                ret = episode_returns(ep, args.gamma, value_fn, args.device)
                for t in ts:
                    states.append(ep.states[t])
                    res_actions.append(ep.action_residual[t])
                    returns.append(ret[t])

            obs = torch.as_tensor(np.stack(states), device=args.device)
            acts = torch.as_tensor(np.stack(res_actions), device=args.device)
            ret_t = torch.as_tensor(np.stack(returns), device=args.device)

            value_pred = value_fn(obs)
            critic_loss = nn.functional.mse_loss(value_pred, ret_t)
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            with torch.no_grad():
                advantage = ret_t - value_fn(obs)
            # weight_clip bounds the WEIGHT itself (AWR/Peng et al. clip it to ~20), not
            # the exponent -- clamping the exponent to +-20 instead (an earlier bug here)
            # lets exp(20) ~ 5e8 through and blows up the actor loss. Clamp the exponent
            # to log(weight_clip) so exp(...) can never exceed weight_clip; no lower
            # clamp needed, exp() underflows to ~0 safely for very negative advantage.
            weight = torch.exp((advantage / args.beta).clamp(max=np.log(args.weight_clip)))
            log_prob = policy.log_prob(obs, acts)
            actor_loss = -(weight * log_prob).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            step += 1
            if step % 20 == 0:
                print(f"[learner] step {step}  critic_loss={critic_loss.item():.4f} "
                      f"actor_loss={actor_loss.item():.4f} adv_mean={advantage.mean().item():.3f} "
                      f"episodes={len(buf.episodes)} transitions={len(buf)}")
            if step % args.save_every == 0:
                # Only one checkpoint file, overwritten in place -- versioned per-step
                # files would just accumulate forever across a long-running online run.
                save_atomic(policy, args.ckpt_dir / "residual_latest.pt", extra={"step": step})
                print(f"[learner] saved checkpoint @ step {step}")

        if time.time() - last_refresh >= args.refresh_interval_s:
            n = buf.refresh()
            last_refresh = time.time()
            if n:
                print(f"[learner] +{n} closed episodes ({len(buf.episodes)} total, {len(buf)} transitions)")


if __name__ == "__main__":
    main()
