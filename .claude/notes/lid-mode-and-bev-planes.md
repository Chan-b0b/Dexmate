# `--lid` mode, BEV warp planes, and depth — findings 2026-09-04

What this session established on the robot, with the numbers, so none of it has
to be re-derived. Code lives in `LGES/ik_demo/` (config split into
`LGES/ik_demo/config/`) and `LGES/case_detection/`.

---

## 1. The BEV warp plane is NOT the surface height (this cost the most)

`bev.build_mapper(q_torso, q_head, plane_z)` warps the head frame to a metric
top-down canvas **on the plane at `plane_z`**. Warping one image at two parallel
planes cuts the same bundle of camera rays at two heights, so the two results
are a **homothety about the camera centre**:

```
P_to = C_xy + (P_from - C_xy) * (z_to - C_z) / (z_from - C_z)
```

(`bev.reproject_plane`, added by the bin-plane work; `bev.camera_centre` gives C.)

Consequences:

* **Angles are plane-invariant** — a wrong plane never costs yaw.
* **A centre detected at the wrong plane is exactly recoverable** — no
  re-detection, no re-warp, just rescale about `C_xy`.
* **Lengths scale by the same factor**, so a feature of known physical size pins
  its own height (`bev.plane_from_size`).

Measured on the unload lid: warped at the configured `0.345` while the surface
was really `0.145` → the reported centre was **175 mm short in x** with a clean,
confident box. Every reach verdict computed from that xy was wrong.

**Rule: the warp plane is a DETECTION choice (the scale the model recognises);
the surface height is a MEASUREMENT.** Keep them as separate numbers. The bin
OBB set was labelled on canvases warped at `top_face_z(1) = 0.6138` (see
`case_detection/calib_bin_plane.py`), so the lid appears 2.4x oversized at
`0.345` and 3.5x at its true `0.145` — moving the warp plane to the "correct"
height would hurt recognition for no geometric gain.

## 2. ZED depth measures the surface, and it is trustworthy

`bot.sensors.head_camera.get_depth()` → `(H, W) float32, metres`, registered to
the left image, **enabled by default** (`enable_depth=True`, zenoh topic
`sensors/head_camera/depth`). At the time of writing rgb and depth were both
`600x960`. Depth is **Z along the optical axis**, the same convention
`camera_geometry.deproject_pixel` inverts.

`case_detection/depth_plane.py` (new):

* `base_to_pixel(base_xyz, q_torso, q_head)` — project a base point to a pixel.
* `sample_depth(depth, rgb_shape, u, v, half_win_px=12, pct=50)` — percentile of
  the valid depth in a window, scaled into the depth map's own resolution.
* `plane_from_depth(...)` — measure the surface height under a base xy, iterating
  so the sampled pixel converges even from a badly wrong guess (synthetic check:
  recovers a plane to **±0.2 mm** from guesses 150 mm off).
* `expected_depth(base_xyz, ...)` — what the map *should* read there. The
  discriminator when a measurement disagrees with config.

**Validated against force**: `lid_probe`'s touch test descended the empty cup and
stopped on force at `ee_z 0.2846` → surface `0.1296`, against `0.1446` from depth
at a nearby spot. Two independent sensors, ~2 cm apart on a lid with 2 cm of
relief; a repeat run agreed to **0.3 mm**. The configured value at the time
(`0.345`) was **20 cm out**.

### Sampling geometry that matters

* The window is **±12 px**, i.e. `12*d/f` on the surface: 36 mm at 0.55 m,
  **49 mm at 0.75 m**, 79 mm at 1.2 m (f = 366.2 px).
* **Percentile matters on a surface with relief.** 50 = the face's height (what
  the reprojection wants). ~10 = the nearest points = the tallest bumps, which is
  what a descending cup lands on first (`cfg.LID_DEPTH_CONTACT_PCT`). Measured
  lids carry ~20 mm of embossing; `p95-p5` is reported as `relief` and reads
  21-24 mm, matching the eye.
* **The contact height is sampled under the CUP** (centre + rotated
  `LID_GRAB_OFFSET_M`), not under the centre — 20 mm away can be ridge vs groove.
* **The first sample pixel walks with the guess.** Plane error `dz` moves it
  `|P - C_xy| * dz / (z - C_z)`: 55 mm → 49 mm, 200 mm → **176 mm**, against a
  0.5x0.35 m lid's 175 mm half-width. 0904's success was luck. Fixed by seeding
  the sampling guess from the last measurement (`_LID_PLANE_SEEN`, keyed per warp
  plane so pick and place do not seed each other).

## 3. `--lid` flow as built

```
operator drives to the lidded box -> `d`
  detect (warp cfg.LID_PLANE_Z_M) -> depth-refine centre + contact height
  wrist-yaw search: canonical, its 180 flip, then outward (cup is round, so yaw
      is free; only the pick COLUMN binds)
  align the chassis ONLY IF no yaw solves the column where the lid is
  pick (creep_gap = the long DESCENT_CREEP_GAP_M: the plane is an estimate)
  lift to SAFE_TRANSPORT_Z   <- no view park: the lift already clears the box by
                                55 mm, a park at 1.08 leaves only 35 mm
operator drives to the unload spot -> `d`
  torso -> LID_PLACE_TORSO_DEG AND both arms -> taught stow joints, TOGETHER
  head -> LID_PLACE_HEAD_PITCH_DEG, detect the FLOOR lid, depth-refine
  align the chassis ONLY IF the place column does not already solve, and then to
      the NEAREST solvable spot, not to a taught point
  place: descend from contact+100 mm, force-stopped, blow-off
  torso -> back to TORSO_JOINTS
```

Key decisions and why:

* **Both chassis legs are hand-driven**; nothing describes those paths.
* **Torso and arms move together** (`lid_place_stance`, via
  `_park_during_legs`) — the torso carries both arm bases, so a lean with frozen
  arms drags the held lid through a pose nobody chose. The arm targets are
  **joint vectors**, because during the lean there is no fixed base to aim a
  Cartesian target at, and the taught configs are on a checked elbow branch.
* **`move_torso` for the motion, `pin_torso` after** — `pin_torso` rebuilds the
  pinocchio model at the end, and doing that under a running arm stream swaps the
  model out from under it.
* **Taught joints are clipped into the IK band** (`ArmMover.clip_to_band`): a
  config read off the robot can sit outside `JOINT_RANGE_FRAC` (the stow's j7 was
  0.4 deg out), which `move_joints` would happily command while the model called
  it invalid.
* **The yaw delta is carried pick → place.** `lid_yaw - wrist_yaw` is invariant
  while held, so `W_place = W_pick + (L_floor - L_pick)`; the place re-adds
  whatever the pick had to rotate by, or the lid lands rotated by exactly that.
  Passing through a fixed joint stow does not break this.
* **Correct only what is broken.** `_center_lid` drives to a reference and knows
  nothing about reach; calling it unconditionally re-parked the robot over spots
  that already solved (0904: a 380 mm-tall reachable band still got corrected),
  and each open-loop correction adds its own error. Both sides now test the
  column first, and the place aims at `_nearest_place_spot` — the run's own case
  went from `+130 mm forward, -340 mm sideways` to `+60/-30 mm`.
* **x safety floors** (`LID_CENTER_MIN_X_M 0.84`, `LID_PLACE_MIN_X_M 0.86`):
  closing on the box/drop-off is the one correction with something in front of
  the robot. Normally redundant (the reference sits on the floor) — that is the
  point.

## 4. Reach envelopes measured (torso-stance specific!)

At `LID_PLACE_TORSO_DEG = (14.8, 59.0, -60.3)` (torso pitch **-14.5 deg**, which
is what `set_head_pitch(angle=...)` is relative to):

| column | reachable |
|---|---|
| 0.45 -> 0.35 | x 0.84-0.95, y 0.15-0.40; **x 1.00 no** |
| 0.385 -> 0.285 (measured) | x 0.84-0.90 for y 0-0.30; x 0.95 needs y>=0.15 |

The pick column at `LID_PLANE_Z_M + cup` is wide open (`x[0.70,1.06]` at
y=+0.05) for every wrist yaw except ~260 deg. **A lid at x~1.05 (where the
depth-corrected detection actually put it) is NOT reachable** — the chassis has
to close in ~15 cm first.

`set_head_pitch(bot, angle=A)` sets `head_j1 = torso_pitch_deg - A` where
`torso_pitch = j1 + j3 + 90 - j2`. At the place stance `A = 24` puts the floor
spot at pixel row 317 of 631 — dead centre; 20-50 all keep it in frame.

## 5. Creep gap: two knobs, because two regimes

`DESCENT_CREEP_GAP_M = 0.05` was sized from the **cup deviation a fast stretch
leaves behind** (40 mm at 0.2 m/s, halving per 11 mm of creep). That no longer
applies to legs whose approach ENDS at rest on the creep line and is confirmed
there by `_settle_at`: 0904 logs show the pick's creep (`seal`) entering at
`|track_xy| = 0.5, 0.3, 0.5 mm`, while the case place (`corner`), which still
streams per-tick from the hover, enters at `3.5 -> 7.9 -> 21.8 mm` and does need
the distance.

So `DESCENT_CREEP_GAP_SETTLED_M = 0.02` for the settled legs (pick, non-corner
place) — **0.75 s saved per leg** — and the full 0.05 stays for the corner seat.
The short gap is an **expected-z error budget**, so a column whose expected z is
a guess passes the long one explicitly (`pick/place(creep_gap=...)`; `--lid`
does).

Raising the creep SPEED does not help: the decay is a time constant (11 mm at
0.04 m/s = 0.275 s half-life), so a faster creep needs a proportionally longer
gap for the same decay and only raises the impact force.

## 6. dexcontrol: 0.4.9 vs 0.5.0

The install moved from site-packages **0.4.9** to an editable **0.5.0** of
`dexcontrol/` mid-session. They differ in ways that matter:

* **0.5.0 has `move_to_joint_pos`** (robot-server motion plugin: trajectory
  smoothing + gravity comp) returning a `MotionHandle` to `wait()` on. 0.4.9 has
  no motion plugin at all.
* **What is deprecated is only `set_joint_pos(wait_time>0)`** — the
  client-side interpolate-and-block mode. `set_joint_pos_vel` and
  `set_joint_pos(wait_time=0)` are the supported STREAMING API and the docstrings
  explicitly require calling them at 100-500 Hz. **The arm must keep streaming**
  (`_send` at `CONTROL_HZ`): per-tick warm IK, force-halt-on-contact, corner
  drive and the velocity feedforward all depend on owning the loop.
* `arm.move_torso()` picks the right call for whatever is installed. On 0.4.x it
  reproduces the 0.5 behaviour by scaling the direction vector
  `set_joint_pos_vel` would otherwise send at the FULL velocity ceiling — that
  full-speed step was the "torso lurches" complaint. `cfg.TORSO_VEL_SCALE = 0.2`.
* `Robot()` construction intermittently died in `_set_default_state`, which reads
  the TORSO state to compensate the head home pose, before that subscriber had a
  parsed sample. `_wait_for_components` only waits for CRITICAL components and has
  already returned, so there is nothing to pre-wait on: `arm.connect_robot()`
  retries the whole construction (`ROBOT_CONNECT_ATTEMPTS/DELAY_S`) and re-raises
  the last failure.

## 7. Tools built for this

* **`ik_demo.jog_ee`** — keyboard EE jog for taking poses. Single keys, no Enter
  (`w/s`=±x, `a/d`=±y, `r/f`=±z, `,/.`=yaw, `-/+`=step, `v`=level the cup,
  `p`=pose, `g`=goto, `t`=torso, `q`). Holds the orientation the arm STARTS in
  (forcing the vertical cup made every key `blocked` from the joint park, whose
  own xy cannot hold the cup vertical at all). Straight-line per-tick streams at
  creep speed; re-orients IN PLACE first, because a leg that also rotates the
  wrist fails its first tick's IK and halts having never moved.
* **`ik_demo.lid_probe`** — detect + reachability + depth, interactive.
  `--setup` reproduces the sequence's own place stance via `lid_place_stance`.
  `d` detect + z band + depth, `m` xy map, `v` **TOUCH TEST** (descend the empty
  cup to real force contact and compare with the prediction — the only check that
  closes the whole chain against a sensor), `a` align if needed / `a!` always,
  `l/r/f/b`+`tl/tr` drive. Saves marker images to `cfg.LID_IMAGE_DIR`.
* **`case_detection/depth_plane.py`** — the depth geometry above.

## 8. Traps worth remembering

* **A long-lived interactive process keeps the modules it imported.** Editing
  `chassis_sequence.py` while `lid_probe` is open changes nothing in it; a touch
  test then silently used config fallbacks. The log wording ("lid detected on 3/3
  frames (plane z=...)" vs "(warped at z=...)") is what caught it. Restart the
  tool after touching the library.
* **Never derive a height by converting a ground-referenced reading.** "0.40
  above the ground" plus a ground height guessed from the URDF wheel axles gave
  0.34; the surface was 0.13. Measure in base_link, or measure the difference to
  something known (the profile: floor `-0.045`, lid `+0.145`, i.e. 0.18 apart).
* **`descent_reachable` checks down to `DESCENT_CHECK_BOTTOM_EE_Z` (0.755)**,
  which is ABOVE the whole lid-place column. Use `column_reachable(hi, lo)` with
  explicit bounds for anything low.
* **The self-collision model puts the OTHER arm at `pin.neutral` = zeros**, a
  pose the robot is never in. At torso stances that swing the arms inward
  (j3 = -60 deg) the zeroed right arm intersects `base_0`, so `in_collision()`
  came back True for EVERY left-arm config and every column pre-check refused.
  `_setup_model` now seeds it from the live joints (its home when headless).
* **`_track_trace`'s `z=` column is the COMMANDED z on planned legs**, not where
  the arm is; the live position is `z + track_z`. A 0.95 read that way was really
  1.05, and reading it as live is how a 198 mm plane error hid for a run.
