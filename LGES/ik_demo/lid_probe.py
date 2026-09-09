"""Detect the floor lid where the robot stands NOW and report what it can reach.

By default no motion except the head pitch: the torso is taken exactly as it is
(the model is built from the live joints). With ``--setup`` it first puts the
robot into the SAME state the place runs in — torso to LID_PLACE_TORSO_DEG and
the arm to LID_UNLOAD_STOW_JOINTS, both at once, via the sequence's own
lid_place_stance() — because a band measured in a different stance measures
nothing. It answers the two questions the --lid place depends on:

  * where is the lid, how repeatable the detection is (spread over N frames),
    and what the ZED DEPTH says the surface height under it actually is —
    the warp plane both lid detections currently take on faith
  * over what z BAND does the cup column actually solve at that xy, per wrist
    yaw — i.e. how high the approach may start and how low the descent may go

and then says whether the configured LID_PLACE_START_EE_Z_M / LID_PLACE_EE_Z_M
fit inside that band, plus a small xy map so a chassis correction has a
direction to go in.

It is interactive: the chassis can be driven from the same prompt (the
move_chassis grammar the run's manual legs use) and every command can be
followed by another `d` to re-measure, so "does this spot work?" is one
keystroke. `a` runs the same alignment the sequence does — including its
"only if the column does not already solve" policy, so the probe cannot
disagree with the run about whether a spot needs correcting; `a!` aligns
regardless, for trying a different spot out.

    python -m LGES.ik_demo.lid_probe
    python -m LGES.ik_demo.lid_probe --n 5        # detection frames (median)
    python -m LGES.ik_demo.lid_probe --no-head    # leave the head alone
    python -m LGES.ik_demo.lid_probe --box-lid     # the PAPER box lid instead
    python -m LGES.ik_demo.lid_probe --setup       # take the place stance first
    python -m LGES.ik_demo.lid_probe --z-top 1.0  # start the sweep higher

An UNREACHABLE IK solve costs ~50ms (it spends the full IK_MAX_ITERS) against
2-3ms for a reachable one, so the sweeps are deliberately bounded: --z-top
caps the column, the band walk stops once it has left a solvable stretch, and
the xy map only tests the CONFIGURED column instead of the whole range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from dexcontrol.core.config import get_robot_config

from . import config as cfg
from .arm import ArmMover, connect_robot
from .suction import SuctionMover
from .chassis_sequence import (_LAST_LID_DIMS, _LAST_LID_FRAME, _box_lid_detector,
                                _center_lid, _detect_lid_xy, _place_column_ok,
                                lid_place_stance, run_stamp, set_head_pitch)

Z_BOTTOM, Z_STEP = 0.10, 0.02


def _rpy(yaw: float):
    return (float(cfg.GRASP_ORIENTATION_RPY[0]),
            float(cfg.GRASP_ORIENTATION_RPY[1]), float(yaw))


def _ok(m: ArmMover, x, y, z, rpy, seed):
    sol = m.solve_pose((float(x), float(y), float(z)), rpy, seed=seed,
                       min_motion=seed is not None)
    good = (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits
            and not sol.in_collision)
    return good, sol


def z_band(m: ArmMover, x: float, y: float, yaw: float, z_top: float):
    """(z_high, z_low) of the FIRST contiguous solvable stretch of the column at
    (x, y), walking down from ``z_top`` warm-chained like the descent pre-check.
    Stops once that stretch ends — the descent needs one continuous column, and
    every failed z costs ~50ms."""
    rpy, seed, run = _rpy(yaw), None, None
    for z in np.arange(z_top, Z_BOTTOM - 1e-9, -Z_STEP):
        good, sol = _ok(m, x, y, z, rpy, seed)
        if good:
            seed = sol.q
            run = (float(z), float(z)) if run is None else (run[0], float(z))
        elif run is not None:
            break                      # left the stretch: that is the band
        else:
            seed = None
    return run


def column_fits(m: ArmMover, x: float, y: float, yaw: float, hi: float, lo: float):
    """Does the CONFIGURED column hi -> lo solve at (x, y)? Returns the lowest z
    reached (None if even ``hi`` fails), so a map cell has a number in it.
    Bails at the first failure — a broken column is broken."""
    rpy, seed, low = _rpy(yaw), None, None
    for z in np.arange(hi, lo - 1e-9, -Z_STEP):
        good, sol = _ok(m, x, y, z, rpy, seed)
        if not good:
            return low
        seed, low = sol.q, float(z)
    return low


def _yaws(base: float):
    """The place's own wrist-yaw candidates: canonical, then the 180 flip."""
    return [float((base + b + np.pi) % (2.0 * np.pi) - np.pi) for b in (0.0, np.pi)]


_SHOT = [0]


def _save_depth_marker(rgb, depth, marks, d_med, d_exp, half_win_px: int = 12):
    """Write the RGB and a colour-mapped DEPTH view with the sampled window
    drawn on both, so "is the window even on the lid?" is answerable by eye.

    ``marks``: [(u, v, label, bgr), ...] in the INTRINSICS frame (~995x631);
    each is scaled into whichever image it is drawn on. The depth view is
    clipped to 0.3-1.2m — the band the lid and the floor both fall in — with
    invalid pixels left black."""
    if cfg.LID_IMAGE_DIR is None:
        return
    try:
        import cv2

        Path(cfg.LID_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        # numbered per shot: the run stamp is fixed for the process, so a bare
        # name meant every `d` overwrote the last one — useless for the compare-
        # two-spots workflow this tool is for
        _SHOT[0] += 1
        base = (Path(cfg.LID_IMAGE_DIR)
                / "run_{}_depth_{:02d}".format(run_stamp(), _SHOT[0]))
        d = np.asarray(depth, dtype=np.float64)
        ok = np.isfinite(d) & (d > 0.05)
        norm = np.clip((d - 0.30) / (1.20 - 0.30), 0.0, 1.0)
        view = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        view[~ok] = 0
        for img, name in ((cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR), "rgb"),
                          (view, "depth")):
            h, w = img.shape[:2]
            for u, v, label, colour in marks:
                su, sv = u * w / 995.0, v * h / 631.0
                r = int(round(half_win_px * w / 995.0))
                cv2.rectangle(img, (int(su - r), int(sv - r)),
                              (int(su + r), int(sv + r)), colour, 2)
                cv2.drawMarker(img, (int(su), int(sv)), colour,
                               cv2.MARKER_CROSS, 22, 2)
                cv2.putText(img, label, (int(su + r + 4), int(sv - r - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            cv2.putText(img, "median {} m / expected {:.3f} m".format(
                "none" if d_med is None else "{:.3f}".format(d_med), d_exp),
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
            cv2.imwrite(f"{base}_{name}.png", img)
        logger.info("saved {}/{}_rgb.png and _depth.png (yellow = the window at "
                    "the CONFIGURED plane, red = after the depth measurement; "
                    "depth view clipped 0.3-1.2m, black = invalid)",
                    cfg.LID_IMAGE_DIR, base.name)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not save the depth marker images: {}", e)


def _depth_plane(bot, cup_xy) -> None:
    """MEASURE the height of the surface under the detected cup point, from the
    ZED depth map, and report it against the configured warp plane.

    The warp plane has to be the height the feature actually lies on; both lid
    planes in config are estimates (one back-computed off an E-stopped run, one
    converted from a ground-referenced reading), and being wrong there scales
    the detected center about the camera nadir — a clean, confident box tens of
    mm out in x. Depth answers it directly, per frame. Report only: nothing
    downstream uses it yet."""
    try:
        import bev
        import depth_plane as dp

        f = dict(_LAST_LID_FRAME)
        rgb, joints, plane = f.get("rgb"), f.get("joints"), f.get("plane_z")
        if rgb is None:
            logger.warning("no stored detection frame — cannot measure the plane")
            return
        depth = bot.sensors.head_camera.get_depth()
        if depth is None:
            # still write the RGB with the window on it: "where would we have
            # sampled" is the next question, and a missing depth stream is
            # exactly when you want to see the frame
            u, v = dp.base_to_pixel((cup_xy[0], cup_xy[1], float(plane)),
                                    joints[0], joints[1])
            _save_depth_marker(rgb, np.zeros((2, 2), dtype=np.float32),
                               [(u, v, "cfg plane {:.3f}".format(float(plane)),
                                 (0, 255, 255))], None, float("nan"))
            logger.warning("no depth frame (stream enabled? topic "
                           "sensors/head_camera/depth) — saved the RGB anyway")
            return
        # raw diagnostics FIRST: when a measurement disagrees with the config,
        # these are what tell a wrong plane from a wrong pixel mapping
        u, v = dp.base_to_pixel((cup_xy[0], cup_xy[1], float(plane)),
                                joints[0], joints[1])
        d_med, n0, sp0 = dp.sample_depth(depth, rgb.shape, u, v)
        d_exp = dp.expected_depth((cup_xy[0], cup_xy[1], float(plane)),
                                  joints[0], joints[1])
        logger.info("DEPTH raw: rgb{} depth{} {} | cup->px ({:.1f},{:.1f}) | "
                    "median {} m vs {:.3f} m expected at the configured plane | "
                    "{} px, spread {:.0f}mm", tuple(rgb.shape[:2]),
                    tuple(depth.shape[:2]), depth.dtype,
                    u, v, "none" if d_med is None else "{:.3f}".format(d_med),
                    d_exp, n0, sp0 * 1000)
        z, n, spread = dp.plane_from_depth(depth, rgb.shape, joints[0], joints[1],
                                           cup_xy, float(plane))
        marks = [(u, v, "cfg plane {:.3f}".format(float(plane)), (0, 255, 255))]
        if z is not None:
            C0 = bev.camera_centre(joints[0], joints[1])
            xy_z = bev.reproject_plane(cup_xy, float(plane), z, C0)
            u2, v2 = dp.base_to_pixel((xy_z[0], xy_z[1], z), joints[0], joints[1])
            marks.append((u2, v2, "measured {:.3f}".format(z), (0, 0, 255)))
        _save_depth_marker(rgb, depth, marks, d_med, d_exp)
        if z is None:
            logger.warning("depth window had only {} valid pixels — no measurement", n)
            return
        C = bev.camera_centre(joints[0], joints[1])
        fixed = bev.reproject_plane(cup_xy, float(plane), z, C)
        logger.info("DEPTH: surface under the cup point is at z={:.4f} "
                    "(configured plane {:.4f}, off by {:+.0f}mm) — {} px, "
                    "spread {:.0f}mm", z, plane, (z - plane) * 1000, n, spread * 1000)
        # A MINI HEIGHT PROFILE around the lid. The absolute value depends on
        # the whole chain (intrinsics, head/torso FK, depth convention); the
        # DIFFERENCES do not, because every sample rides the same chain. So
        # "lid minus the floor beside it" is the number to compare against a
        # tape measure, and it is what tells a genuinely low lid from a depth
        # chain that over-reads.
        prof = []
        for dxo, dyo, nm in ((0.0, 0.0, "cup"), (0.0, +0.25, "left"),
                             (0.0, -0.25, "right"), (+0.25, 0.0, "far"),
                             (-0.20, 0.0, "near")):
            zz, nn, ss = dp.plane_from_depth(depth, rgb.shape, joints[0], joints[1],
                                             (cup_xy[0] + dxo, cup_xy[1] + dyo),
                                             float(plane))
            prof.append("{} {}".format(nm, "n/a" if zz is None
                                       else "{:+.4f}".format(zz)))
        logger.info("DEPTH profile (z at 25cm around the cup point): {}",
                    "  ".join(prof))
        logger.info("DEPTH: correcting for that plane moves the cup point "
                    "({:.4f},{:+.4f}) -> ({:.4f},{:+.4f}), i.e. ({:+.0f},{:+.0f})mm",
                    cup_xy[0], cup_xy[1], fixed[0], fixed[1],
                    (fixed[0] - cup_xy[0]) * 1000, (fixed[1] - cup_xy[1]) * 1000)
        # A lid with relief has no single height: the cup lands on the TALLEST
        # bump under it, so contact comes from the nearest depth in the window
        # (p10), not the median. The two differ by roughly the relief.
        z_hi, _, _ = dp.plane_from_depth(depth, rgb.shape, joints[0], joints[1],
                                         cup_xy, float(plane), pct=10.0)
        logger.info("DEPTH: relief — face (median) z={:.4f}, tallest points "
                    "(p10) z={:.4f}, {:.0f}mm apart", z,
                    z if z_hi is None else z_hi,
                    0.0 if z_hi is None else (z_hi - z) * 1000)
        logger.info("DEPTH: the cup would first touch at ee_z {:.4f} (tallest + "
                    "cup) / {:.4f} (median + cup) — configured LID_PLACE_EE_Z_M "
                    "{:.3f}", (z if z_hi is None else z_hi) + cfg.SUCTION_LENGTH_M,
                    z + cfg.SUCTION_LENGTH_M, cfg.LID_PLACE_EE_Z_M)
    except Exception as e:  # noqa: BLE001 — a measurement must not stop the probe
        logger.warning("depth plane measurement failed: {}", e)


def _touch_forces():
    """(contact, abort) thresholds for the touch test — the gentle pair on the
    PAPER lid, the demo's own on the bin lid. Cardboard dents under the 10N
    global gate: 0905's first probe read 11.3N before it stopped."""
    if _TARGET.get("box"):
        return float(cfg.BOX_LID_CONTACT_N), float(cfg.BOX_LID_FORCE_LIMIT_N)
    return float(cfg.FORCE_CONTACT_THRESHOLD_N), float(cfg.FORCE_HARD_LIMIT_N)


def _touch_test(m: SuctionMover, det, sz: float, pz: float) -> None:
    """Descend the EMPTY cup onto the lid and compare where it actually touched
    with where depth said it would.

    This is the only check that closes the whole chain — intrinsics, head/torso
    FK, the depth convention, the warp plane, the reprojection — against a
    sensor that cannot be argued with. Everything else in this tool is a
    prediction. Force-guarded and creep-speed, the same descent the pick flies;
    no suction, so the cup just rests on the lid and lifts off."""
    cup = (det[3], det[4])
    wyaw = float(np.deg2rad(det[2])) + float(cfg.GRASP_YAW)
    rpy = None
    for b in (0.0, np.pi):
        cand = float((wyaw + b + np.pi) % (2.0 * np.pi) - np.pi)
        r = (float(cfg.GRASP_ORIENTATION_RPY[0]),
             float(cfg.GRASP_ORIENTATION_RPY[1]), cand)
        if m.column_reachable(cup[0], cup[1], r, sz, pz):
            rpy = r
            break
    if rpy is None:
        logger.error("no wrist branch solves the column {:.3f} -> {:.3f} at "
                     "({:.3f},{:+.3f}) — cannot touch-test here", sz, pz, *cup)
        return
    touch_n, abort_n = _touch_forces()
    logger.warning("TOUCH TEST: the cup will descend onto the lid at "
                   "({:.3f},{:+.3f}), wrist {:+.1f}deg, from ee_z {:.3f}, "
                   "expecting contact at {:.4f}; contact gate {:.1f}N, abort "
                   "{:.1f}N. E-stop in reach.", cup[0], cup[1],
                   float(np.rad2deg(rpy[2])), sz, pz, touch_n, abort_n)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return
    q = m._approach_and_hover((cup[0], cup[1], pz), rpy, pz, to_creep_z=True,
                              tick_cb=m._approach_force_guard(abort_n),
                              approach_z=sz, creep_gap=cfg.DESCENT_CREEP_GAP_M)
    if q is None:
        logger.error("approach unreachable — nothing moved")
        return
    res = m._descend_to_contact(pz, rpy, abort_n, q, contact_n=touch_n)
    if res.contact_ee_z is None:
        logger.error("no contact ({}) — the cup did not reach the surface", res.reason)
    else:
        logger.warning("TOUCH TEST: contact at ee_z {:.4f} vs {:.4f} predicted "
                       "({:+.1f}mm) — reason {}", res.contact_ee_z, pz,
                       (res.contact_ee_z - pz) * 1000, res.reason)
        logger.info("TOUCH TEST: that puts the surface at z={:.4f} (depth said "
                    "{:.4f}); config LID_PLACE_EE_Z_M would have predicted {:.3f}",
                    res.contact_ee_z - cfg.SUCTION_LENGTH_M,
                    pz - cfg.SUCTION_LENGTH_M, cfg.LID_PLACE_EE_Z_M)
    m.move_ee_vertical(sz, rpy)          # straight back up, xy held


_TARGET = {"detector": None, "plane": None, "name": "bin lid", "box": False}


def _detect(bot, n):
    """Detect + the cup point, or None.
    (lid x, lid y, yaw_deg, cup x, cup y, measured surface z | None) — the last
    one is what cfg.LID_DEPTH_REFINE measured, and it is what `v` and the band
    checks descend to; None means depth did not answer and the configured
    fallback applies."""
    det = _detect_lid_xy(bot, n=n, plane_z=_TARGET["plane"],
                         detector=_TARGET["detector"])
    if det is None:
        logger.error("no {} detected (warp plane z={:.3f}) — check the head pitch "
                     "and that it is in frame", _TARGET["name"], _TARGET["plane"])
        return None
    if _LAST_LID_DIMS:
        logger.info("detected size (long x short): {:.3f} x {:.3f} m, conf {:.2f}",
                    _LAST_LID_DIMS.get("long", 0.0), _LAST_LID_DIMS.get("short", 0.0),
                    _LAST_LID_DIMS.get("conf", 0.0))
    lx, ly, lyaw_deg = det[0], det[1], det[2]
    lyaw = float(np.deg2rad(lyaw_deg))
    ox, oy = cfg.LID_GRAB_OFFSET_M
    c, s = float(np.cos(lyaw)), float(np.sin(lyaw))
    cup = (lx + c * ox - s * oy, ly + s * ox + c * oy)
    logger.info("lid center ({:.4f},{:+.4f}) yaw {:+.1f}deg -> cup point "
                "({:.4f},{:+.4f})", lx, ly, lyaw_deg, *cup)
    _depth_plane(bot, cup)
    return lx, ly, lyaw_deg, cup[0], cup[1], (det[3] if len(det) > 3 else None)


def _bands(m, cup, lyaw_deg, z_top, want_hi, want_lo) -> None:
    logger.info("--- reachable z band at the DETECTED cup point (sweep from "
                "{:.2f}) ---", z_top)
    for yaw in _yaws(float(np.deg2rad(lyaw_deg)) + float(cfg.GRASP_YAW)):
        b = z_band(m, cup[0], cup[1], yaw, z_top)
        if b is None:
            logger.warning("  wrist {:+7.1f}deg: NO solvable z at all",
                           float(np.rad2deg(yaw)))
            continue
        fits = b[0] >= want_hi - 1e-6 and b[1] <= want_lo + 1e-6
        logger.info("  wrist {:+7.1f}deg: z {:.3f} down to {:.3f} ({:.0f}mm tall) "
                    "— configured column {}", float(np.rad2deg(yaw)), b[0], b[1],
                    (b[0] - b[1]) * 1000, "FITS" if fits else "does NOT fit")


def _map(m, cup, lyaw_deg, want_hi, want_lo) -> None:
    logger.info("--- lowest z the CONFIGURED column ({:.2f} -> {:.2f}) reaches, "
                "best of the two wrist branches ---", want_hi, want_lo)
    yaws = _yaws(float(np.deg2rad(lyaw_deg)) + float(cfg.GRASP_YAW))
    logger.info("        " + "".join(f"  y{cup[1] + dy:+.2f}"
                                     for dy in (-0.10, -0.05, 0.0, 0.05, 0.10)))
    for dx in (-0.10, -0.05, 0.0, 0.05, 0.10):
        cells = []
        for dy in (-0.10, -0.05, 0.0, 0.05, 0.10):
            lows = [column_fits(m, cup[0] + dx, cup[1] + dy, w, want_hi, want_lo)
                    for w in yaws]
            lows = [v for v in lows if v is not None]
            cells.append("   --  " if not lows else
                         " {:.2f}{}".format(min(lows),
                                            "* " if min(lows) <= want_lo + 1e-6
                                            else "  "))
        logger.info("  x{:+.2f} {}", cup[0] + dx, "".join(cells))
    logger.info("(cell = lowest cup z the column reaches; `*` = the whole "
                "configured column solves there, `--` = not even the start)")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="detection frames (median)")
    ap.add_argument("--no-head", action="store_true", help="do not move the head")
    ap.add_argument("--box-lid", action="store_true",
                    help="probe the PAPER box lid (box OBB class "
                         "cfg.BOX_LID_CLS_ID, warped at cfg.BOX_LID_PLANE_Z_M) "
                         "instead of the bin lid — for taking its z")
    ap.add_argument("--setup", action="store_true",
                    help="first take the place stance (torso + arm together), as "
                         "the --lid sequence does, instead of measuring where the "
                         "robot happens to be")
    ap.add_argument("--target", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="cup xy the `a` alignment drives to (default "
                         "LID_PLACE_XY_M)")
    ap.add_argument("--z-top", type=float, default=0.80,
                    help="top of the z-band sweep (default 0.80; nothing above "
                         "that solves at the lid-place stance)")
    args = ap.parse_args()

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        torso = np.asarray(bot.torso.get_joint_pos(), dtype=float)
        logger.info("live torso {} rad = {} deg  (cfg.LID_PLACE_TORSO_DEG = {})",
                    np.round(torso, 4), np.round(np.rad2deg(torso), 1),
                    cfg.LID_PLACE_TORSO_DEG)
        # SuctionMover, not ArmMover: `v` descends to real force contact, which
        # needs the wrench reference and the guarded descent. Nothing here turns
        # the vacuum on.
        _TARGET.update(
            detector=(_box_lid_detector if args.box_lid else None),
            plane=float(cfg.BOX_LID_PLANE_Z_M if args.box_lid
                        else cfg.LID_FLOOR_PLANE_Z_M),
            name=("paper box lid" if args.box_lid else "bin lid"),
            box=bool(args.box_lid))
        logger.info("probing the {} (warp plane {:.3f}{})", _TARGET["name"],
                    _TARGET["plane"],
                    ", box OBB class {}".format(cfg.BOX_LID_CLS_ID)
                    if args.box_lid else "")
        m = SuctionMover(bot)            # model at the LIVE torso
        if args.setup:
            logger.warning("--setup: moves the TORSO to {} deg and the ARM to the "
                           "stow joints, together — clear the robot",
                           cfg.LID_PLACE_TORSO_DEG)
            if input("Continue? [y/N]: ").strip().lower() != "y":
                return
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            # ready at the LIVE torso: ensure_ready would drag it to the demo
            # stance first, which is the opposite of what this is for
            if not m.ensure_ready_live_torso(release_estop=release):
                logger.error("arm not ready — aborting")
                return
            lid_place_stance(bot, m)
        elif np.max(np.abs(torso - np.deg2rad(
                np.asarray(cfg.LID_PLACE_TORSO_DEG, dtype=float)))) > 0.05:
            logger.warning("the live torso is NOT the place stance — every band "
                           "below is measured in the stance the robot is in now. "
                           "Re-run with --setup to take the place stance first.")
        if not args.no_head:
            set_head_pitch(bot, angle=cfg.LID_PLACE_HEAD_PITCH_DEG)
        logger.info("head {} deg", np.round(np.rad2deg(
            np.asarray(bot.head.get_joint_pos(), dtype=float)), 1))

        want_hi = float(cfg.LID_PLACE_START_EE_Z_M)
        want_lo = float(cfg.LID_PLACE_EE_Z_M)
        logger.info("configured column: start {:.3f} -> expected contact {:.3f}",
                    want_hi, want_lo)
        target = (tuple(float(v) for v in cfg.LID_PLACE_XY_M)
                  if args.target is None else tuple(args.target))

        from .move_chassis import (move_backward, move_forward, strafe_left,
                                   strafe_right, turn_ccw, turn_cw)
        moves = {"l": strafe_left, "r": strafe_right,
                 "f": move_forward, "b": move_backward}
        turns = {"tl": turn_ccw, "tr": turn_cw}

        det = _detect(bot, args.n)
        if det is not None:
            _bands(m, det[3:5], det[2], float(args.z_top), want_hi, want_lo)
        def _heights(t):
            """(start, contact) for a detection — measured when depth refined it,
            the configured pair otherwise. The same rule run_lid uses, so the
            touch test descends to exactly what the run would."""
            if t is None or len(t) < 6 or t[5] is None:
                return want_hi, want_lo
            c = float(t[5]) + float(cfg.SUCTION_LENGTH_M)
            return c + (want_hi - want_lo), c

        logger.info("commands: d=detect+bands  m=xy map  v=TOUCH TEST (descend to "
                    "real contact)  a=align IF NEEDED to ({:.3f},{:+.3f})  "
                    "a!=align anyway  l/r/f/b [m] [speed]  tl/tr [deg] [rad_s]  "
                    "q=quit", *target)
        while True:
            try:
                parts = input("probe> ").strip().lower().split()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not parts:
                continue
            cmd = parts[0]
            if cmd == "q":
                break
            if cmd == "d":
                det = _detect(bot, args.n)
                if det is not None:
                    _bands(m, det[3:5], det[2], float(args.z_top), want_hi, want_lo)
            elif cmd == "m":
                if det is None:
                    logger.warning("detect first (`d`)")
                else:
                    _map(m, det[3:5], det[2], *reversed(_heights(det)))
            elif cmd == "v":
                if det is None:
                    logger.warning("detect first (`d`)")
                else:
                    hi, lo = _heights(det)
                    _touch_test(m, det, hi, lo)
            elif cmd in ("a", "a!"):
                # Same policy as the sequence: only correct when the column does
                # NOT already solve where the lid is. `a!` forces the move, for
                # trying a spot out. Unconditional alignment was the earlier
                # behaviour and it re-parked the robot over spots the probe had
                # just called FITS — exactly the discrepancy this tool exists to
                # rule out.
                if det is None:
                    logger.warning("detect first (`d`)")
                elif cmd == "a" and _place_column_ok(m, det[:3], 0.0,
                                                    want_hi, want_lo):
                    logger.info("already reachable — not moving (`a!` aligns anyway)")
                else:
                    got = _center_lid(bot, plane_z=_TARGET["plane"],
                                      x_ref=target[0], y_ref=target[1],
                                      x_min=float(cfg.LID_PLACE_MIN_X_M))
                    if got is None:
                        logger.warning("alignment gave up (no/rejected detection)")
                    det = _detect(bot, args.n)
                    if det is not None:
                        _bands(m, det[3:5], det[2], float(args.z_top),
                               want_hi, want_lo)
            elif cmd in moves or cmd in turns:
                try:
                    a = float(parts[1]) if len(parts) > 1 else None
                    sp = float(parts[2]) if len(parts) > 2 else None
                    (moves[cmd] if cmd in moves else turns[cmd])(
                        bot, **({"distance_m": a} if cmd in moves
                                else {"angle_deg": a}), speed=sp)
                except (ValueError, IndexError) as e:
                    logger.warning("parse error: {} — l/r/f/b [m] [speed] | "
                                   "tl/tr [deg] [rad_s]", e)
            else:
                logger.warning("commands: d  m  a  a!  l/r/f/b [m] [speed]  "
                               "tl/tr [deg] [rad_s]  q")


if __name__ == "__main__":
    _main()
