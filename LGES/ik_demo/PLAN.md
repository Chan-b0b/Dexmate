# ik_demo — rebuild plan

Clean reimplementation of `case_battery_demo`. Same physical task (suction
pick-and-place of a case + two batteries, with barcode-matched batteries
diverted to the right-hand gripper), rebuilt for clarity, stability, and speed.

## Scope

**In**
- IK + arm motion core (hardened pink + Ruckig)
- Suction pick / place
- Barcode scan + divert
- Gripper handoff (right-arm Robotiq)
- Pose teaching

**Out (deliberately dropped)**
- Undo / reverse sequence  → removes `run_undo`, `Move.reversed`, `_done`, recorded pick-z
- `--loop`                 → meaningless without undo
- VLA episode recording    → removes recorder hooks threaded through the orchestrator
- Live dashboard / telemetry → removes `_publish_*` spool writes, head-camera enable juggling
- Episode XY shift         → VLA-diversity only
- Z-stacking repeats (`FORWARD_REPEATS`, `Z_STEP_PER_REPEAT`)

## Forward task (only direction)

```
1. case      : CASE_PICK   -> CASE_PLACE_R
2. battery_1 : BAT_SRC_1    -> BAT_SLOT_1     (barcode gate: match -> divert to gripper)
3. battery_2 : BAT_SRC_2    -> BAT_SLOT_2     (barcode gate: match -> divert to gripper)
```
Each move: pick -> lift to transport z -> travel -> place (or handoff on divert).

## IK / motion strategy — the core rebuild

Problem with the old approach: an **online, local differential-IK solver** on a
redundant 7-DOF arm. Symptoms in the old code: branch flips mid-lift (patched by
pinning the posture target to the start config), ill-conditioning at the
cross-body reach (jitter had to be disabled — it amplified into ~10 rad/s
commands), solve-as-you-go with no pre-execution validation, and conservative
scattered speed caps to stay stable. Slow *because* only stable when slow.

**Decisions**

1. **Fixed poses are solved once, not live.** The 6 taught poses never change.
   Solve each to a joint vector offline, curate the branch (away from
   limits/singularities), validate, and cache. Runtime interpolates between
   known-good joint configs — deterministic, repeatable, no branch-flip lottery.

2. **Live IK is confined to the 3 sensing legs only:**
   - descend-to-contact (suction pick/place seal)
   - barcode spiral search (x/y waypoints on a no-read)
   - handoff grip (target computed from the live suction EE pose + offset)

3. **Ruckig for all time-parameterization.** One kinematic budget
   (`MAX_*_VEL / ACCEL / JERK`) drives every move; duration derives from path
   length. "Make it faster" = raise those numbers, and motion stays jerk-limited
   and feasible everywhere. Replaces the scattered `MOVE_DURATION_S` /
   `LIFT_AVG_SPEED_M_S` / per-leg duration constants.
   - fixed-pose -> fixed-pose travel: Ruckig in **joint space** between two
     cached configs (safe by construction, no live IK).
   - descent / vertical clearance / sensing legs: Ruckig in **Cartesian
     space**, per-setpoint IK, warm-started.

4. **Validate before streaming.** For any Cartesian leg with live IK, generate
   the whole trajectory first and gate on: all substeps converged, inter-step
   joint jump below threshold, within limits. Abort *before* the arm moves if it
   fails. (Joint-space Ruckig legs between validated endpoints are safe by
   construction.)

5. **Adaptive damping (stretch):** scale `lm_damping` by manipulability so the
   solver stiffens near singularities — lets the sensing legs run faster safely.

6. **Exactly two motion primitives — no `lift`, no `move_to`.**
   - `move_joints(q)` — no-IK joint-space Ruckig. The workhorse: cached-pose ->
     cached-pose travel, plus home / taught joint poses.
   - `move_ee(pos, rpy)` — live-IK Cartesian Ruckig, for the sensing legs only.
     Vertical clearance is just `move_ee` with x, y, rpy held (a straight
     Cartesian line that only varies z), so the old `lift` is subsumed;
     `move_to` was only `move_ee` to transport-z + a settle wait, so it's gone
     too. No settle-poll primitive: Ruckig trajectories end at rest, leaving at
     most a brief guard before a contact-sensitive leg.
   - **Redundancy resolution differs by phase** (this is what made `lift` a
     separate method in the old code — now split cleanly): offline pose-solve
     pulls toward joint mid-ranges (curate away from limits); live `move_ee`
     minimizes joint motion from the start config (stays on one branch, no flips).

### Descent-to-contact (pick & place — the one non-precomputable leg)

Contact height isn't known until the wrench/vacuum fires, so descent stays a
sensing loop, not a precomputed trajectory. Rebuild:

- **One primitive** `descend_until(guards, floor, speed)`; pick vs place differ
  only in the guard set + force limit. Replaces the old approach_to_contact /
  _descent_loop / _descend_to triplication. Drop `_descent_loop` (already dead —
  pick now uses approach+seal) and jitter (disabled / documented harmful).
- **Hybrid: fast Ruckig approach + admittance contact (RECOMMENDED, confirm at
  step 4).**
  - *Free-air*, down to `contact_z + 5 cm`: fast Ruckig position descent via
    `set_joint_pos_vel` (velocity feedforward cancels tracking lag; removes the
    old open-loop "lead the lagging arm + `_halt()`" hack). **5 cm gap for now** —
    taught contact z is fixed but possibly inaccurate; re-measure against
    real-deploy wrench readings and tighten later.
  - *Contact zone* (final creep + touch + seal-press): switch to **admittance
    control** — the arm behaves as a spring-damper, so contact force is bounded
    by design (≈ stiffness × penetration), robust to an inaccurate taught z, and
    holds a gentle bounded press while the vacuum seals. Detect-and-freeze
    (position + force-threshold halt) is the fallback if admittance tuning is
    fiddly.
  - Adoption cost is low: the `AdmittanceController` law is ~90 lines of pure
    numpy (see `dexcontrol/examples/advanced_examples/admittance_control.py`) —
    `dexmotion`/`pytransform3d` are NOT needed (not installed); feed it the arm's
    `wrench_sensor.get_wrench_state()` (6-vector) + our FK pose, run the corrected
    pose through our pink IK. Latency-sensitive → keep it on the Thor.
- **Keep (physics doesn't change):** two-signal pick (force = contact, DI0 =
  seal) with approach → pre-lift → suction-on → seal-wait; per-descent tare
  (empty for pick, battery-loaded for place) + post-tare sanity check; two force
  limits (hard = abort, contact = stop; place tighter).
- Warm-started IK makes 100 Hz feasible (tiny per-tick z delta → ~1 solver
  iteration); benchmark a warm solve on the Thor when building `arm.py`.

## Module layout

```
ik_demo/
  run_demo.py     # entry point: CLI, robot lifecycle, orchestrator
  config.py       # slim config (one place; see below)
  arm.py          # ArmMover: pink IK setup/solve + Ruckig trajectory execution
                  #           + fixed-pose solve/validate/cache
  suction.py      # SuctionMover: descend-to-contact, seal, force limits, place
  barcode.py      # scan-gate descend + spiral search + target-match divert
  gripper.py      # GripperMover: Robotiq control + handoff choreography
  sequence.py     # TaskOrchestrator: forward moves, retry, divert branching
  teach.py        # pose teaching (EE + joint), merges the two old teach tools
  poses/          # taught_*.txt, ee_place_seq.txt, cached joint_targets.json
  drivers/        # reused-as-is validated drivers (do not rewrite)
    suction_io.py #   HTTP weblogic + VacuumMonitor (DI0)
    bcr.py        #   Cognex DataMan telnet scanner
    robotiq.py    #   Robotiq Modbus gripper
```
`read_force` (wrist-wrench contact detection) stays an external import from
`grasp_box`, as today.

## Reuse vs rewrite

- **Reuse as-is** (hardware-validated, high risk to rewrite): `drivers/*`
  (suction_io, bcr, robotiq), external `read_force`, and the pink IK
  setup/solve math ported cleanly into `arm.py`.
- **Rewrite clean**: `ArmMover` structure + Ruckig execution, `sequence.py`
  orchestration, `config.py`, `run_demo.py`, `teach.py`.

## Config — slim target

Keep: IK params (frames, orientations, damping, convergence), kinematic budget
(`MAX_JOINT_VEL/ACCEL/JERK`, `MAX_EE_VEL/ACCEL/JERK`), suction descent + force
thresholds, barcode config, Robotiq + handoff config, taught poses,
`SAFE_TRANSPORT_Z`, hover/approach heights.

Drop: `PHASE_INSTRUCTIONS`, `EPISODE_XY_SHIFT_MAX_M`, `FORWARD_REPEATS`,
`Z_STEP_PER_REPEAT`, jitter block, dashboard/trace-for-dashboard bits, the fixed
per-leg duration constants (replaced by the kinematic budget).

## Build order (bottom-up, each testable before the next lands on it)

1. **Prereq**: ~~`pip install ruckig`~~ DONE — ruckig 0.17.3 in `/opt/venv`.
2. **drivers/**: copy suction_io, bcr, robotiq unchanged; smoke-test each.
3. **arm.py**: IK setup/solve; fixed-pose solve+validate+cache tool; the two
   primitives — `move_joints` (joint-space Ruckig between cached configs) and
   `move_ee` (Cartesian live-IK, min-motion redundancy). Verify: move between
   all taught poses, no branch flips, single speed knob.
4. **suction.py**: descend-to-contact, seal, force limits, place. Verify a pick.
5. **sequence.py**: forward case + 2 batteries (no barcode/gripper yet). Verify.
6. **barcode.py**: scan-gate + spiral + divert flag. Verify a read + a no-read.
7. **gripper.py**: Robotiq + handoff choreography. Verify a full divert.
8. **run_demo.py**: CLI + lifecycle. Verify end-to-end.

## Open questions

- **Torso pitch dependency (found during arm.py build).** Reachability of the
  taught base_link poses depends on the torso pitch (`torso_j2`) — it moves the
  arm base. Solving at torso=0 leaves every pose ~300 mm unreachable (this, not
  a dexmotion bug, was the earlier "IK fails" symptom; both our pink and
  dexmotion failed identically at torso=0). So the pose cache is only valid at
  the torso angle the poses were taught at. On-robot, `ArmMover` reads the live
  torso and it just works. Consequences: (1) `teach.py` must record the torso
  angle alongside each pose; (2) the cache must be stamped with its torso angle
  and re-solved if the runtime torso differs; (3) the old `taught_*.txt` files
  do NOT FK-match this model at any torso angle, so treat them as stale — re-teach.
- Cache invalidation: when taught poses change, the joint cache must be
  re-solved. Plan: `teach.py` re-solves + revalidates the affected pose on save.

## Build status
- Done: drivers/, config.py, arm.py (model + own pink IK + pinocchio/SRDF
  self-collision + Ruckig joint trajectories + set_joint_pos_vel streaming +
  pose-cache/validation + headless self-test). dexmotion NOT used at runtime.
- **arm.py VALIDATED headless (`python -m ik_demo.arm`):** all 6 taught poses
  solve to <1 mm, converged, collision-free, in-limits; warm solve 0.82 ms
  (100 Hz confirmed); Ruckig joint trajectory generated. Keys: torso at
  `cfg.TORSO_JOINTS` = deg[80,175,25]; IK seeded from `HOME_JOINTS_*` and
  warm-chained (a zero seed stalls ~340 mm short).
- arm.py move_joints streaming verified on the robot (smooth). Tuned:
  SPEED_SCALE=0.8, SUCTION_LENGTH_M=0.25 (cup-tip offset via `taught_target`).
- suction.py (step 4) WRITTEN — stage 1 detect-and-freeze: per-tick IK streamed
  via set_joint_pos_vel (finite-diff vel feedforward), tared native-wrench
  vertical force = contact, DI0 = seal; two-speed descent (fast + creep), two
  hard-force limits, pre-lift seal, blow release. Force via arm.wrench_sensor
  (no read_force/grasp_box). Imports OK. On-robot test: `python suction.py
  [POSE]` (home -> pick -> lift).
- Deferred to stage 2 (after stage-1 verified): admittance contact zone; a
  Cartesian-straight move_ee (current _move_ee_to is joint-space point-to-point).
- suction.py pick VERIFIED on robot (contact 11.4N @ ee_z 0.834, sealed, lifted).
  Fix that made it work: taught poses are already L_gripper_base frame — do NOT
  add SUCTION_LENGTH (was double-counting). pick() now lifts to transport on
  success so pick/place compose.
- sequence.py (step 5) WRITTEN — Move(label,src,dst) + FORWARD_MOVES (case,
  battery_1, battery_2); TaskOrchestrator.run_forward = per-move pick(src) ->
  place(dst) with config-gated retry (MAX_PHASE_ATTEMPTS). `_run_move` is the
  barcode/gripper divert hook. On-robot: `python sequence.py`. Imports OK.
- Next: verify sequence.py on robot, then barcode.py (step 6) + gripper.py (7),
  then run_demo.py (8).
