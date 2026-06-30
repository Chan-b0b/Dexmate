# MPC — latent dynamics model + planning for the LGES task

A TD-MPC2-style **latent world model** trained on the case+battery demo data,
to be used for **deploy-time MPC** (sampling-based planning at decision time).

## Decisions (2026-06-22)

- **Goal:** deploy-time planning / MPC, using a learned latent world model.
- **State representation:** the existing 15-dim low-dim state
  (`ee pos(3) + quat(4) + suction(1) + sealed(1) + raw wrench(6)`), *not* images.
  It contains everything the descend/seal/lift dynamics and the MPC objective
  depend on, and keeps latent rollouts cheap enough for per-tick planning at
  15 Hz. Images can be added as a second encoder branch later if needed.
- **Reward (for the later planner stage):** analytic, built from the per-sub-task
  target poses in `config.py` + a seal/release bonus + a force penalty. No reward
  learning (demo data is success-only, so there is little reward signal to learn).
- **Milestone 1 (this code): dynamics model + validation only.** No planner yet —
  there is no point building MPPI on a model that cannot predict the seal event.

## Key structural fact

The recorded action is the observed next-state delta, so
`pos[t+1]=pos[t]+dpos`, `quat[t+1]=drot∘quat[t]`, `suction[t+1]=action[6]` are
**exact**. The only genuinely *learned* dynamics are **wrench(6)** and the
**seal event(1)** — contact forces and when suction grabs/releases. The
validation harness scores those hardest and compares against a persistence
baseline.

## Files

### Stage 1 — dynamics model (validated)
- `data.py` — loads only the low-dim parquet columns (images untouched),
  normalizes (z-scores pos+wrench+action deltas; leaves quat/flags raw), builds
  multi-step sub-trajectory windows.
- `model.py` — `LatentDynamics` (encoder → latent dynamics → decoder, SimNorm
  latent) + multi-step rollout loss (decode + latent-consistency + reconstruction).
- `train.py` — offline training loop, cosine LR, saves best checkpoint + norm stats.
- `validate.py` — multi-step open-loop prediction error vs. horizon, seal-toggle
  timing, persistence baseline, and per-episode rollout plots.

Result: from horizon k≥5 the model beats a persistence baseline by a widening
margin on wrench + seal (k=50/3.3 s: fz 1.8 N vs 3.1 N, seal acc 0.98 vs 0.80).
Latent pos decode is imperfect (~10 mm) → the planner integrates pose analytically.

### Stage 2 — MPC planner (verified in imagination)
- `reward.py` — analytic reward with per-phase contact targets **derived from
  data** (median EE pose at the seal/release toggle). Reach-to-contact is always
  active; the seal/release bonus is **gated by proximity to the contact pose**.
- `planner.py` — `MPPIPlanner`: sampling-based MPC over the latent model. Pose is
  integrated analytically from the sampled action deltas; the model supplies the
  seal probability and contact force the reward needs. Receding horizon with
  warm-start; same `state[15] → action[7]` interface as `run_policy.py`.
- `evaluate_planner.py` — offline verification: action-agreement vs. demos +
  imagined closed-loop (planner drives the model's own predictions) + plots.

**Model-exploitation finding (important):** the first reward (bonus on raw
`sealed_prob`) was immediately reward-hacked — MPPI found off-distribution
actions where the model wrongly predicts a seal in mid-air, so the planner
"sealed" at hover and never descended. Fix: make reach-to-contact always-on and
**gate the seal/release bonus by proximity to the object** (a mid-air seal is
physically meaningless). After the fix, imagined rollouts are correct: pick
descends ~29 cm and seals *at contact*; place descends and *releases at the slot*.

This is verified in **imagination only** (planner vs. the model's own
predictions) — it confirms reward+planner+model compose into the right
primitives, but does NOT prove real-world success. Single-model MPC can still be
exploited in ways imagination can't reveal; the robustness items below address that.

### Stage 3 — on-robot deployment (wired; not yet tested on hardware)

`LGES/vla_training/run_policy.py` gained an `--mpc` flag: it swaps the SmolVLA
policy for `MPPIPlanner` as the per-tick action source and **skips the cameras**
(MPC needs only the 15-dim state), re-grounding the reference to the live pose
every tick (`chunk_steps=1`). All existing guards are reused unchanged: per-step
dpos/drot clamps, workspace box, force-limit abort, IK joint-jump abort,
seal-based task termination, ENTER-abort, and retreat-to-hover on any non-clean
stop. Deploy params `H=15 N=256 iters=3` plan in ~24 ms (warmed up) — fits 15 Hz.

## Run (use the vla_venv python)

```bash
/home/dexmate/vla_venv/bin/python MPC/train.py            # -> MPC/runs/dyn/best.pt
/home/dexmate/vla_venv/bin/python MPC/validate.py         # dynamics-model metrics + plots
/home/dexmate/vla_venv/bin/python MPC/evaluate_planner.py # planner verification + plan_*.png
```

On-robot (from `LGES/vla_training/`, needs the robot + a human on the e-stop):
```bash
# 1) DRY-RUN first — prints planned actions + IK feasibility, COMMANDS NOTHING:
python run_policy.py --mpc --dry-run --task case_pick --goto-start <case_pick_take_dir>
# 2) GO — commands the arm; start conservative (tight force limit, box on):
python run_policy.py --mpc --go --task case_pick --goto-start <take_dir> --box --force-limit 8
# chain: --chain case_pick case_place ...   single sub-task: --task <phase>
```

Data: `LGES/vla_training/datasets/lges_suction` (train) / `_val` (held out).

## Next

1. **Dynamics ensemble + disagreement penalty** — the principled fix for model
   exploitation (penalize reward where ensemble members disagree = off-manifold).
2. **SmolVLA as action prior** — seed MPPI sampling from the policy so search
   stays near in-distribution actions; directly attacks the limit cycle.
3. Reward/value head + terminal-value bootstrap (full TD-MPC2) to extend horizon.
4. Lift-after-seal at deploy: the reward drives descend+seal/release but not the
   lift; run_policy's seal-based termination handles the pick/place boundary, but
   a dedicated "lift to hover once the *real* seal is observed" target switch
   would be cleaner (don't trust the model's seal for the phase switch).
5. On-robot test (wired via `run_policy.py --mpc`): dry-run, then `--go` from a
   `--goto-start` pose; collect targeted off-manifold data around the limit-cycle
   failure region to close the success-only data gap.
