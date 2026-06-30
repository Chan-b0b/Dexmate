# On-robot executor — design doc (for review)

First closed-loop run of the trained SmolVLA policy. **Scope: one task,
`case_pick`, left arm, suction only, operator watching.** Nothing here
chains tasks or touches the right arm/gripper. Goal of this first run is
not a perfect pick — it's to see the policy drive the arm safely and
sanely, and to surface the gap between the 1.77 mm open-loop fit and
closed-loop reality.

Decisions locked with the user: reuse the demo's `Mover` IK/servo
primitives (closest to training-time dynamics, least new code); single
task; this doc before any code.

## 1. Control loop (target ~15 Hz, matching the recorder's sample rate)

```
setup:  bring arm to a known case_pick start pose via existing primitive
        (Mover.goto / taught pose) — NOT under policy control
        policy.reset(); suction_off()

each tick (dt = 1/15 s):
  1. obs   = build_observation()          # must match recorder byte-for-byte (§2)
  2. act   = policy.select_action(obs)    # 7-dim: dpos(3) drot(3) suction(1)
  3. act   = safety_clamp(act)            # §4
  4. tgt   = integrate(current_ee_pose, act)   # §3
  5. tgt   = workspace_clamp(tgt)         # §4
  6. q,ok  = mover._solve_ik(fresh_cfg, tgt.pos, tgt.rpy)
             if not ok: hold / abort (§4)
  7. mover._arm.set_joint_pos(arm_q_from(q))
  8. suction_on()/off() per act[6] threshold, via suction_io
  9. check aborts (force, workspace, IK, dead-man, max-steps) → stop
  sleep to maintain dt
```

Why command the **integrated absolute** pose through IK each tick (not feed
deltas to a velocity controller): the recorder logged absolute EE pose and
the action is `pose[t+1] − pose[t]`, so integrating deltas onto the live
pose and re-solving IK reproduces exactly what the data represents. One IK
solve + one `set_joint_pos` per tick; the 15 Hz cadence *is* the
interpolation (no inner easing like `_move_to_joints`).

## 2. Observation — must match the recorder exactly

The #1 failure mode is an observation the policy never saw in training.
The recorder's `states.jsonl` + converter define the contract:

| state dim | source | notes |
|-----------|--------|-------|
| 0–2 pos (x,y,z) | publisher FK in `base_link`, `cfg.EE_FRAME` (`L_gripper_base`) | **use the same FKHelper the publisher uses**, not `Mover`'s pinocchio FK, unless they're verified identical |
| 3–6 quat (w,x,y,z) | same FK, rpy→quat | converter applies sign-continuity *within an episode*; at deploy we just need a consistent hemisphere — match the training normalization, see open Q1 |
| 7 suction | commanded suction bit (`is_suction_commanded_on()`) | 1.0/0.0 |
| 8–13 wrench | `arm.wrench_sensor.get_state()["wrench"][:6]` raw fx,fy,fz,tx,ty,tz | **raw**, never tared |

Image: head RGB → resize to 512×320 → RGB order → the preprocessor pipeline
(rename head→camera1, normalize, the model resizes to 256). Reuse
`convert_to_lerobot.py`'s exact resize so train and deploy match.

The cleanest way to guarantee parity: import the publisher's FK + wrench
read, or factor the recorder's frame-build into a shared helper both call.
**Proposed:** a small `obs_common.py` that both the recorder and executor
import, so they can never drift. (Refactor is optional for the first run;
if we skip it, the executor must copy the publisher logic verbatim.)

## 3. Action decode & integration

```
dpos  = act[0:3]                       # metres, base frame
drot  = act[3:6]                       # rotvec, base frame: R_delta = exp(drot)
suction = act[6] > 0.5

tgt.pos  = cur.pos + dpos
R_tgt    = R_delta @ R_cur             # left-multiply: delta is in base frame
tgt.rpy  = matrix_to_rpy(R_tgt)        # _solve_ik takes rpy
```

Note the converter built `drot` as `rotvec(q_{t+1} q_t^{-1})` — a base-frame
(left) rotation — so integration must left-multiply. This must be verified
against the converter math before running (open Q2).

## 4. Safety layer (every clamp is a hard requirement for run #1)

1. **Per-step delta clamp** — clip `dpos` and `drot` to the training action
   range (from the saved normalizer: dx ±11 mm, dy ±45 mm, dz [−78,+30] mm,
   rot ±~50 mrad). A single out-of-distribution spike can't jump the arm.
   Start *tighter* than the data (e.g. 50%) for the very first run.
2. **Workspace box** — clamp integrated `tgt.pos` to an explicit
   `[x,y,z]` min/max box around the known case-pick region. Hard-stop (not
   clamp-and-continue) if the *unclamped* target leaves the box by more than
   a margin — that means the policy diverged.
3. **Force hard limit** — abort if `raw_mag` (or vertical force via
   `read_force`) exceeds `cfg.FORCE_HARD_LIMIT_N` (20 N). Reuses the demo's
   existing limit.
4. **IK failure** — if `_solve_ik` returns `ok=False`, hold the last good
   `q` for that tick; abort after N consecutive failures.
5. **Dead-man + e-stop** — the loop only steps while the operator holds a
   key (or: runs for a fixed small number of ticks then stops for inspection).
   Any key / Ctrl-C → `suction_off()` + hold. First runs should be
   "hold-to-run", not "press-to-start-and-walk-away".
6. **Max steps** — hard episode cap (e.g. 1.5× the longest training
   case_pick episode) so a non-terminating policy can't run forever.
7. **Known start pose** — always begin from the same taught pose the
   operator used during recording; never hand the policy control from an
   arbitrary configuration.

Abort = `suction_off()`, stop commanding, leave arm where it is, print why.

## 5. Termination / success

For run #1, success detection is optional — operator calls it. If we want
auto-stop: reuse the vacuum-seal signal (`VacuumMonitor` /
`VACUUM_SEAL_TIMEOUT_S`) the demo already uses, i.e. stop when sealed.
Decision deferred (open Q3) — manual stop is fine to start.

## 6. Open questions (need answers before coding the run)

- **Q1 — quat hemisphere at deploy.** Training applied per-episode sign
  continuity. At deploy we have a single live quat per tick. Need to confirm
  the policy is robust to quat sign (state normalization should handle it,
  but worth a 1-line check: does flipping the sign of a sample's quat change
  the predicted action?). Low risk, must verify.
- **Q2 — delta rotation frame/convention.** Confirm left-multiply (base
  frame) matches the converter's `quat_mul(q[t+1], conj(q[t]))` exactly,
  including wxyz ordering, before any motion.
- **Q3 — auto success-stop** vs manual: manual for run #1 unless you want
  the seal-based stop wired in now.
- **Q4 — FK parity.** Is the publisher's FKHelper numerically identical to
  `Mover`'s pinocchio FK for `L_gripper_base`? If yes, simplest to use
  `Mover` for both obs and IK. If not, obs must use the publisher's FK.

## 7. Files to add (none of this runs the robot on its own)

```
LGES/vla_training/
  run_policy.py        # the executor (this design)
  obs_common.py        # OPTIONAL shared frame-builder (recorder + executor)
```

`run_policy.py` will require an explicit `--task case_pick` and an explicit
`--arm-it` / `--go` flag to move; default dry-run prints predicted actions
and clamps without commanding the arm, so we can watch the policy "think"
before it touches anything.

## 8. Recommended bring-up order

1. **Dry run** — loop reads real obs, prints predicted Δ + clamped target +
   IK feasibility, commands nothing. Confirms obs parity and sane outputs.
2. **Suspended / cleared workspace** — first real motion with the arm clear
   of the case, hold-to-run, tight clamps. Watch for sane trajectory.
3. **Real case_pick** — full task, normal clamps, operator on the e-stop.
```
