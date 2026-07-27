# Collision

Model-based collision detection for the Vega left arm.

## Files

- `collision_monitor.py` — two-layer background monitor comparing measured
  joint current against a calibrated gravity + Coulomb-friction model.
  Layer A (change of residual over 0.1 s) catches impacts and is robust to
  grasped objects and calibration drift; Layer B (absolute banded residual)
  catches slow pushes and stalls — after grasping call
  `monitor.retare_payload()` (or `set_extra_payload(kg)`) to re-baseline it.
  Freezes both arms on trigger. Thresholds default to the
  `*_suggested_thresholds` in `calibration_left.json`.
  Standalone: `python collision_monitor.py [--dry-run]`.
- `calibrate_gravity_model.py` — identifies the real EE payload mass
  (gripper + suction, not in the URDF), per-joint torque→current gains, and
  Coulomb friction by sweeping designated poses (`CALIB_POSE_OFFSETS`)
  forward and back while sampling during motion (static holds are
  stiction-dominated and irreproducible), then validates the residual with a
  free wiggle and writes `calibration_left.json`.
- `demo_move_left_ee.py` — random-motion demo: moves the left EE to random
  X/Y/Z targets around its start pose with the monitor armed. On contact the
  motion stops and the sensed force is printed (joint residual [A], torque
  equivalent [Nm], wrist wrench [N]), then the arm holds for `--pause`
  seconds (default 1 s) and resumes; 'q'+Enter or Ctrl+C quits. Also
  exports the shared `LeftArmIK` helper.

## Workflow

```bash
python calibrate_gravity_model.py     # 1. calibrate (~2 min; do this after any tool change)
python collision_monitor.py --dry-run # 2. optional: push the arm, watch residuals
python demo_move_left_ee.py           # 3. random motion with collision stops
```

Calibrate from a pose away from joint limits (e.g. `default_pose.py`), with
the arm's workspace clear. The `--pose-scale 0.5` flag shrinks the sweep for
a cautious first run.

Thresholds are only valid for the motion envelope they were validated on:
the calibration's validation phase runs random 3-D strokes — match its
`--amp-x/y/z` and `--joint-speed` to what your application actually does,
and keep the demo/application speed at or below that value.

## Tuning

- `thresholds` (Layer B, absolute) / `change_thresholds` (Layer A, impact):
  per-joint arrays in the calibration signal unit (normally A); default to
  the JSON suggestions (1.5x the free-motion floor).
- `change_window` (default 0.1 s): longer catches slower impacts, reacts later.
- `n_consecutive` (default 2 polls at 50 Hz): raise to 3 for fewer false
  positives before touching thresholds; each +1 adds 20 ms latency.
- `enable_change` / `enable_absolute`: turn a layer off entirely.
- After grasping/releasing an object: `retare_payload()` (arm still, ~0.5 s)
  or `set_extra_payload(kg)` — otherwise Layer B drifts by ~1 A per kg held
  (Layer A is unaffected, <0.03 A per kg).
- Sensitivity in Newtons ≈ threshold / k_j / (Nm per N at the contact point);
  with the current calibration the shoulder joints detect roughly 5–10 N.
