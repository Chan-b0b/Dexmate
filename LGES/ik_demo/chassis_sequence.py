"""Chassis-based, detection-driven pick & place.

Loops over LAYERS until the source stack is exhausted: each layer runs the full
item set (case -> battery_1 -> battery_2), then the stack heights step
automatically (source -1, target +1) so the BEV warp plane stays on the true
top face. cfg.SRC/TGT_LAYERS_REMAINING are only the STARTING heights.

Per item, one chassis round trip:

    strafe LEFT  (source ~ robot center, open-loop)
      -> detect the case (BEV) -> resolve_poses(detected) -> pick this item
      -> park the arm (clear the head-camera view) IN PARALLEL with the next
         strafe, joined before anything needs the camera (_park_during_legs)
    strafe RIGHT (target ~ robot center)
      -> place at the target:
             run's FIRST case -> the target is empty by definition, so case
                          detection is SKIPPED (only the source stack would be
                          in view); seed the stack at the default front pose,
                          bin-anchored under --auto-move
             otherwise -> detect the case, place aligned to it (same
                          case-frame offset); a miss is a DETECTION failure ->
                          re-detect once, then operator prompt (never
                          blind-stack)
    strafe LEFT  (back, for the next item)

The chassis strafe is OPEN-LOOP (move_sideways = speed*time, no odometry); a
fresh BEV detection recenters the case in base_link at every visit, so the
imprecise strafe is fine. z is not taken from detection — descend-to-contact
finds the real grab/seat height.

Optional flags (sequence.py parity):

    --gripper    battery picks scan the barcode during the descent (pick_gated);
                 a TARGET_BARCODES match is handed to the right-arm gripper at
                 the TARGET position (after the strafe right) instead of being
                 seated in the case. Before the handoff the chassis aligns to
                 the detected divert bin (detect_bin; bin center -> base_link
                 y = DIVERT_BIN_TARGET_Y_M, fixed-strafe fallback if no bin)
                 and strafes back after. The right-arm PLACE runs
                 SYNCHRONOUSLY (a mid-place strafe would drag it through the
                 bin); the right-arm home + left view park afterwards overlap
                 the return strafes (_park_during_legs).
    --auto-move  chassis legs run automatically: fixed CHASSIS_AUTO_STRAFE_DIST_M
                 strafes (overrides CHASSIS_MANUAL). At every station visit the
                 chassis first CENTERS the detected case — turn to case yaw
                 0 deg, strafe so the ITEM's grab/seat point sits on the center
                 line (CHASSIS_CENTER_CASE_Y_M, per-item ref) — before
                 computing the pick/place pose; a failed reach pre-check
                 additionally auto-adjusts from the detection (turn +
                 translate, CHASSIS_ADJUST_* limits) and falls back to the
                 interactive keyboard prompt after CHASSIS_ADJUST_MAX_ATTEMPTS.
                 Arrival residuals (deliberate item re-alignments excluded)
                 feed learned PER-DIRECTION leg distances for the rest of the
                 run (ChassisNav; final values logged at run end).

Run as a PACKAGE so the ik config and the case_detection config don't collide
on the shared name `config`:

    python -m LGES.ik_demo.chassis_sequence [--gripper] [--auto-move] [--dashboard]

Needs a trained BEV detector (case_detection cfg.OBB_MODEL_PATH).
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as cfg
from .barcode import is_target
from .config import resolve_poses
from .gripper import GripperMover
from .suction import SuctionMover
from .drivers import suction_io

# Detection lives in the sibling case_detection package (flat imports). We run in
# package mode so ik uses `from . import config` (-> LGES.ik_demo.config); adding
# case_detection to the path lets its flat `import config`/`import bev` resolve to
# ITS OWN modules without clashing.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "case_detection"))
sys.path.insert(0, str(_HERE.parents[1] / "perception"))
import detect_case_bev as dcb  # noqa: E402
from utils import set_head_pitch  # perception/utils (path added above)  # noqa: E402


# item label -> pose key in resolve_poses (case-frame offset, same for pick@source
# and place@target — only the detected center differs).
ITEMS: tuple[tuple[str, str], ...] = (
    ("case", "CASE_PICK"),
    ("battery_1", "BAT_SRC_1"),
    ("battery_2", "BAT_SRC_2"),
)


def _head_rgb(bot, fresh: bool = True, timeout_s: float = 3.0):
    """Head-camera RGB for detection. get_obs returns the LATEST frame the
    zenoh subscriber has — which can predate the end of a chassis move by the
    pipeline latency + frame interval. A stale frame sees the case where it
    was DURING the strafe, which read as arrival overshoot, crept the robot
    leftward every visit, and fed the creep into the learned leg distance.
    With `fresh` (default), wait until TWO new frames (timestamp changes,
    clock-free) arrive after this call starts — both are then guaranteed
    captured after the chassis stopped. Timeout falls back to the latest
    frame with a warning; non-zenoh transports (no timestamp) skip the wait."""
    def _grab():
        obs = bot.sensors.head_camera.get_obs(obs_keys=["left_rgb"],
                                              include_timestamp=True)
        rgb = obs.get("left_rgb")
        if isinstance(rgb, dict):
            return rgb.get("data"), rgb.get("timestamp")
        return rgb, None
    rgb, ts = _grab()
    if not fresh or rgb is None or ts is None:
        return rgb
    deadline = time.monotonic() + timeout_s
    ticks, last = 0, ts
    while ticks < 2:
        if time.monotonic() >= deadline:
            logger.warning("fresh-frame wait timed out after {} new frame(s) — "
                           "using the latest anyway", ticks)
            break
        time.sleep(0.03)
        rgb2, ts2 = _grab()
        if rgb2 is not None and ts2 is not None and ts2 != last:
            rgb, last = rgb2, ts2
            ticks += 1
    return rgb


def _joints(bot):
    return (np.asarray(bot.torso.get_joint_pos(), dtype=np.float64),
            np.asarray(bot.head.get_joint_pos(), dtype=np.float64))


class ZTracker:
    """Measured-contact z feedforward across layers (chassis port of
    sequence.py's TaskOrchestrator._record_z / _predicted_z).

    The BEV warp plane and the descent's expected z both come from the model
    stack height (FLOOR_Z_BASE_M + layers*LAYER_PITCH_M), which drifts from
    the real stack as layers accumulate — 0804 layer 5 contacted ABOVE the
    creep line, and a wrong plane also biases the detected XY along the
    camera ray. Instead, anchor each (station, label) column on its FIRST
    measured contact ee-z and predict later layers by stepping LAYER_PITCH_M
    from the anchor. The anchor never moves, so one misaligned seat can't
    corrupt later predictions — a deviating contact is flagged instead.
    `layers` is the CURRENT stack height, so one formula serves the shrinking
    source and the growing target.

    With `log_path`, every event also lands in a per-run CSV (anchor / contact /
    misalign / plane / pick_* / place_* failures) — the layer-by-layer error
    data in one place, separate from the run log."""

    def __init__(self, log_path: "str | None" = None) -> None:
        self._anchors: dict[tuple[str, str], tuple[float, int]] = {}
        self._csv = None
        if log_path is not None:
            try:
                p = Path(log_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                self._csv = p.open("w", buffering=1)  # line-buffered: rows survive a crash
                self._csv.write("time,event,station,label,layers,measured_m,predicted_m,resid_mm\n")
                logger.info("z-track CSV: {}", p)
            except OSError as e:  # unwritable dir must not block the run
                logger.warning("z-track CSV disabled ({})", e)

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
            self._csv = None

    def log_event(self, event: str, station: str, label: str, layers: int,
                  measured: "float | None" = None, predicted: "float | None" = None,
                  resid: "str | None" = None) -> None:
        """One CSV row. `measured`/`predicted` are ee-z (or plane-z for the
        'plane' event: measured anchor plane vs model plane). `resid` overrides
        the auto mm residual (non-metric rows: yaw degrees, aspect ratio)."""
        if self._csv is None:
            return
        if resid is None:
            resid = ("" if measured is None or predicted is None
                     else f"{(measured - predicted) * 1000.0:+.1f}")
        self._csv.write(f"{time.strftime('%H:%M:%S')},{event},{station},{label},{layers},"
                        f"{'' if measured is None else format(measured, '.4f')},"
                        f"{'' if predicted is None else format(predicted, '.4f')},{resid}\n")

    def predict_ee_z(self, station: str, label: str, layers: int) -> "float | None":
        a = self._anchors.get((station, label))
        if a is None:
            return None
        z0, l0 = a
        return z0 + (layers - l0) * cfg.LAYER_PITCH_M

    def plane_z(self, station: str, layers: int) -> "float | None":
        """Warp-plane override from the CASE anchor (battery detections also
        detect the case, so the plane is always the case's). Same semantics as
        the model plane: top face at the current stack height — for a target
        CASE place that is one pitch above the existing stack (the plane
        doubles as the place-z model), matching top_face_z(tgt_layers)."""
        z = self.predict_ee_z(station, "case", layers)
        return None if z is None else z - cfg.SUCTION_LENGTH_M

    def place_expectation(self, station: str, label: str,
                          layers: int) -> "tuple[float | None, float | None]":
        """(expected ee-z, misseat tolerance) for a place. Own anchor first; a
        battery column without one borrows a first-place expectation so its
        first contact isn't trusted blindly (0805 L1 battery_2): the sibling
        battery's anchor (symmetric seats, looser tol), else the case's (the
        case grab face is the battery compartment — a battery seat sits at
        most one battery thickness above the case seat)."""
        pz = self.predict_ee_z(station, label, layers)
        if pz is not None:
            return pz, cfg.PLACE_MISSEAT_TOL_M
        if label.startswith("battery"):
            for st, lb in self._anchors:
                if st == station and lb.startswith("battery") and lb != label:
                    logger.info("[{}] {} first place: expectation borrowed from {} "
                                "(tol {:.0f}mm)", label, station, lb,
                                cfg.PLACE_MISSEAT_TOL_SIBLING_M * 1000.0)
                    return (self.predict_ee_z(station, lb, layers),
                            cfg.PLACE_MISSEAT_TOL_SIBLING_M)
            cz = self.predict_ee_z(station, "case", layers)
            if cz is not None:
                logger.info("[{}] {} first place: expectation borrowed from the case "
                            "seat (tol {:.0f}mm = max battery thickness)", label,
                            station, cfg.BATTERY_OVER_CASE_MAX_M * 1000.0)
                return cz, cfg.BATTERY_OVER_CASE_MAX_M
        return None, None

    def record(self, station: str, label: str, layers: int, z: "float | None") -> None:
        """Record a measured contact ee-z: first contact anchors the column,
        later ones are compared against the prediction (misalign flag); a
        contact well BELOW the prediction replaces the anchor (lower = truth)."""
        if z is None:
            return
        pred = self.predict_ee_z(station, label, layers)
        if pred is None:
            self._anchors[(station, label)] = (float(z), int(layers))
            logger.info("[{}] {} z anchored: ee_z={:.4f} @ {} layers",
                        label, station, z, layers)
            self.log_event("anchor", station, label, layers, z)
            return
        z0, l0 = self._anchors[(station, label)]
        if layers != l0:
            # measured contact k layers from the anchor -> the IMPLIED pitch.
            # This is how LAYER_PITCH_M gets calibrated after a case change:
            # read the `pitch` CSV rows of one run (resid = implied - config
            # in mm/layer) and set the config to their mean.
            implied = (float(z) - z0) / (layers - l0)
            logger.info("[{}] {} implied layer pitch {:.1f}mm (config {:.1f}mm)",
                        label, station, implied * 1000.0,
                        cfg.LAYER_PITCH_M * 1000.0)
            self.log_event("pitch", station, label, layers, implied, cfg.LAYER_PITCH_M)
        resid = z - pred
        band = cfg.LAYER_MISALIGN_FRAC * cfg.LAYER_PITCH_M
        if resid < -band:
            # The descent stops at the FIRST thing it touches — it can't read
            # below the real surface, so a LOWER contact means the anchor was
            # from a high (rim) contact: the lower reading is the truth.
            self._anchors[(station, label)] = (float(z), int(layers))
            logger.warning("[{}] {} contact z={:.4f} is {:.1f}mm BELOW the prediction "
                           "{:.4f} — old anchor was a high (rim?) contact, re-anchored",
                           label, station, z, -resid * 1000.0, pred)
            self.log_event("reanchor", station, label, layers, z, pred)
        elif resid > band:
            logger.warning("[{}] {} contact z={:.4f} vs predicted {:.4f} ({:+.1f}mm) — "
                           "possible misalignment", label, station, z, pred, resid * 1000.0)
            self.log_event("misalign", station, label, layers, z, pred)
        else:
            logger.info("[{}] {} contact z={:.4f} ({:+.1f}mm vs predicted)",
                        label, station, z, resid * 1000.0)
            self.log_event("contact", station, label, layers, z, pred)


def detect(bot, layers_remaining: int, plane_z: "float | None" = None):
    """One BEV case detection at the current chassis position, warped at the
    plane for `layers_remaining` (source = full stack, target = built-up).
    `plane_z` overrides the model plane with a measured anchor (ZTracker)."""
    rgb = _head_rgb(bot)
    if rgb is None:
        return None
    return dcb.detect_case_bev(rgb, *_joints(bot), layers_remaining=layers_remaining,
                               plane_z=plane_z)


def _refine_det(bot, layers: int, det, plane_z: "float | None" = None):
    """Median-of-N refinement of the FINAL detection a pick/place pose is
    computed from (cfg.DETECT_MEDIAN_SAMPLES; 1 = off). `det` is the sample
    already in hand (post-centering); N-1 more fresh-frame detections are
    taken and x/y/yaw combined by MEDIAN — robust to single-frame OBB jitter
    and one bad fit (no effect on systematic bias). Yaw samples are unwrapped
    onto the first sample's 180-deg branch before the median (OBB long-axis
    ambiguity flips near the boundary); z comes from the warp plane and is
    identical across samples. Not-found extra samples are dropped; centering
    rounds stay single-shot."""
    n = int(cfg.DETECT_MEDIAN_SAMPLES)
    if n <= 1 or det is None or not det.found:
        return det
    xs, ys, yaws = [det.base_xy[0]], [det.base_xy[1]], [det.base_yaw_deg]
    for _ in range(n - 1):
        d = detect(bot, layers, plane_z)
        if d is None or not d.found:
            continue
        xs.append(d.base_xy[0])
        ys.append(d.base_xy[1])
        yaw = d.base_yaw_deg
        if yaw - yaws[0] > 90.0:      # unwrap onto the first sample's branch
            yaw -= 180.0
        elif yaw - yaws[0] < -90.0:
            yaw += 180.0
        yaws.append(yaw)
    if len(xs) == 1:
        return det
    refined = dataclasses.replace(
        det, base_xy=(float(np.median(xs)), float(np.median(ys))),
        base_yaw_deg=float(np.median(yaws)) % 180.0)
    logger.info("detection refined over {} samples: xy=({:.3f},{:+.3f}) yaw={:.1f}deg "
                "(spread x {:.0f} / y {:.0f} mm)",
                len(xs), refined.base_xy[0], refined.base_xy[1], refined.base_yaw_deg,
                (max(xs) - min(xs)) * 1000, (max(ys) - min(ys)) * 1000)
    return refined


def _dual_plane_probe(bot, layers: int, plane_z: "float | None",
                      station: str, label: str, zt: "ZTracker | None") -> None:
    """One frame, two warps: detection xy with the measured plane vs the model
    plane. The hand-tuned place offsets (PLACE_X_LAYER_TRIM_M, taught poses)
    were tuned against the MODEL plane, so the systematic xy shift the plane
    override introduces is exactly the re-tuning target — this logs it as data
    (CSV rows dual_x/dual_y: measured-plane vs model-plane coordinate).
    Costs one frame grab + one extra inference per item; diagnosis only."""
    if zt is None or plane_z is None:
        return
    rgb = _head_rgb(bot)
    if rgb is None:
        return
    q_torso, q_head = _joints(bot)
    dm = dcb.detect_case_bev(rgb, q_torso, q_head, layers_remaining=layers,
                             plane_z=plane_z)
    d0 = dcb.detect_case_bev(rgb, q_torso, q_head, layers_remaining=layers)
    if not (dm.found and d0.found):
        logger.warning("[{}] {} dual-plane probe: detection missing (measured={} "
                       "model={}) — no shift sample", label, station, dm.found, d0.found)
        return
    dx = (dm.base_xy[0] - d0.base_xy[0]) * 1000.0
    dy = (dm.base_xy[1] - d0.base_xy[1]) * 1000.0
    logger.info("[{}] {} dual-plane shift: measured-model = ({:+.1f}, {:+.1f})mm "
                "(plane {:+.1f}mm)", label, station, dx, dy,
                (plane_z - dcb.bev.top_face_z(layers)) * 1000.0)
    zt.log_event("dual_x", station, label, layers, dm.base_xy[0], d0.base_xy[0])
    zt.log_event("dual_y", station, label, layers, dm.base_xy[1], d0.base_xy[1])
    dyaw = (dm.base_yaw_deg - d0.base_yaw_deg + 90.0) % 180.0 - 90.0  # [0,180) wrap
    zt.log_event("dual_yaw", station, label, layers, dm.base_yaw_deg, d0.base_yaw_deg,
                 resid=f"{dyaw:+.2f}deg")


def _log_det(zt: "ZTracker | None", station: str, label: str, layers: int, det) -> None:
    """CSV rows for the FINAL (pose) detection: det_yaw (raw [0,180) yaw,
    wrapped [-90,90) yaw as used by _center_from_det, conf) and det_box (BEV
    long/short px, aspect ratio). Yaw feeds the slot rotation at the place —
    a yaw error displaces the two battery slots in OPPOSITE x (±0.08m · δ) —
    and an aspect near 1.0 makes the OBB long-axis (yaw) flip-prone."""
    if zt is None or det is None or not det.found:
        return
    yaw = det.base_yaw_deg
    wrapped = yaw - 180.0 if yaw >= 90.0 else yaw
    zt.log_event("det_yaw", station, label, layers, yaw, wrapped,
                 resid=f"conf={det.conf:.2f}")
    long_px, short_px = det.dims_px
    zt.log_event("det_box", station, label, layers, long_px, short_px,
                 resid=f"ar={long_px / max(short_px, 1e-6):.2f}")


def _center_from_det(det) -> tuple[float, float, float, float]:
    """CaseBEV -> resolve_poses source center (x, y, z_EE, yaw_rad).
    z_EE = top-face base z + EE->cup-tip offset.

    Resolve the OBB long-axis 180-deg ambiguity: a top-down suction grasp is
    symmetric under a 180-deg case flip, so wrap the detected yaw to [-90, 90)
    around the taught reference (case yaw 0). A ~180-deg-flipped detection
    otherwise sends the grasp wrist to an unreachable branch (confirmed via
    reach_sweep: at yaw~5.0 rad the reachable x window collapses to ~[0.96,1.01]).
    """
    x, y = det.base_xy
    z_ee = det.top_face_z + cfg.SUCTION_LENGTH_M
    yaw_deg = det.base_yaw_deg
    if yaw_deg >= 90.0:
        yaw_deg -= 180.0
    return (x, y, z_ee, float(np.deg2rad(yaw_deg)))


def _manual_strafe(bot, direction: str) -> bool:
    """Interactive chassis leg (cfg.CHASSIS_MANUAL): drive with `l/r/f/b [dist_m]
    [speed]` / `tl/tr [deg] [rad_s]` commands (move_chassis.py grammar), `d` done.
    Returns False if the user gives up with `q`."""
    from .move_chassis import (strafe_left, strafe_right, move_forward,
                               move_backward, turn_ccw, turn_cw)
    moves = {"l": strafe_left, "r": strafe_right, "f": move_forward, "b": move_backward}
    turns = {"tl": turn_ccw, "tr": turn_cw}
    logger.info("MANUAL chassis leg -> go {} : `l/r/f/b [dist_m] [speed]`, "
                "`tl/tr [deg] [rad_s]`, `d` = in position, `q` = give up", direction.upper())
    while True:
        parts = input(f"chassis[{direction}]> ").strip().lower().split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "d":
            return True
        if cmd == "q":
            return False
        try:
            a = float(parts[1]) if len(parts) > 1 else None
            speed = float(parts[2]) if len(parts) > 2 else None
            if cmd in moves:
                moves[cmd](bot, distance_m=a, speed=speed)
            elif cmd in turns:
                turns[cmd](bot, angle_deg=a, speed=speed)
            else:
                logger.warning("commands: l/r/f/b [dist_m] [speed], tl/tr [deg] [rad_s], "
                               "d = done, q = give up")
        except (ValueError, IndexError) as e:
            logger.warning("parse error: {} — l/r/f/b [dist] [speed] | tl/tr [deg] [rad_s]", e)


def strafe(bot, direction: str, auto: bool = False,
           leg: "ChassisNav | None" = None) -> bool:
    """One chassis leg toward `direction`: fixed-DISTANCE automatic leg when
    `auto` (--auto-move, overrides CHASSIS_MANUAL; `leg` carries the learned
    per-direction distances), manual (interactive) when cfg.CHASSIS_MANUAL,
    else the fixed open-loop speed*time strafe.
    Returns False only when the user gives up a manual leg (`q`)."""
    if auto:
        from .move_chassis import strafe_left, strafe_right
        fn = strafe_left if direction == "left" else strafe_right
        fn(bot, distance_m=leg.dist(direction) if leg is not None
           else cfg.CHASSIS_AUTO_STRAFE_DIST_M,
           speed=cfg.CHASSIS_LEG_SPEED_MS)   # long legs only — corrections stay slow
        return True
    if cfg.CHASSIS_MANUAL:
        return _manual_strafe(bot, direction)
    v = cfg.CHASSIS_STRAFE_SPEED_MS if direction == "left" else -cfg.CHASSIS_STRAFE_SPEED_MS
    logger.info("chassis strafe {} ({:.2f} m/s x {:.1f}s)", direction, v, cfg.CHASSIS_STRAFE_TIME_S)
    bot.chassis.move_sideways(v, wait_time=cfg.CHASSIS_STRAFE_TIME_S)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    return True


def _auto_adjust(bot, det, ref: tuple) -> float:
    """One open-loop chassis correction from a detection whose resolved pose
    failed the reach pre-check (--auto-move): turn in place so the detected case
    yaw matches the taught reference (yaw 0, same [-90,90) wrap as
    _center_from_det), then drive forward/back + strafe so the case center
    (rotated into the post-turn base frame) lands on the reference xy
    (`ref` = SOURCE_CASE_CENTER at the source, TARGET_DEFAULT_CASE_CENTER at
    the target). Per-attempt motion is clamped (CHASSIS_ADJUST_MAX_*) and
    micro-moves skipped (CHASSIS_ADJUST_MIN_*); the caller re-detects and
    re-checks reach after every call.

    Returns the APPLIED lateral move in m (+left; 0.0 if under the deadband)
    so the caller can feed it into the learned leg distances (ChassisNav)."""
    from .move_chassis import (move_backward, move_forward, strafe_left,
                               strafe_right, turn_ccw, turn_cw)
    x, y = det.base_xy
    yaw = det.base_yaw_deg
    if yaw >= 90.0:                 # long-axis 180-deg wrap, as _center_from_det
        yaw -= 180.0
    turn = 0.0
    if abs(yaw) >= cfg.CHASSIS_ADJUST_MIN_TURN_DEG:
        turn = float(np.clip(yaw, -cfg.CHASSIS_ADJUST_MAX_TURN_DEG,
                             cfg.CHASSIS_ADJUST_MAX_TURN_DEG))
        (turn_ccw if turn > 0 else turn_cw)(bot, angle_deg=abs(turn))
    # case center in the post-turn base frame (frame rotated CCW by `turn`)
    th = float(np.deg2rad(turn))
    xr = float(np.cos(th) * x + np.sin(th) * y)
    yr = float(-np.sin(th) * x + np.cos(th) * y)
    lim = cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M
    dx = float(np.clip(xr - ref[0], -lim, lim))   # +: case too far ahead -> forward
    dy = float(np.clip(yr - ref[1], -lim, lim))   # +: case too far left  -> strafe left
    moved_dy = 0.0
    if abs(dx) >= cfg.CHASSIS_ADJUST_MIN_TRANSLATE_M:
        (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx))
    if abs(dy) >= cfg.CHASSIS_ADJUST_MIN_TRANSLATE_M:
        (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy))
        moved_dy = dy
    logger.info("auto-adjust: case ({:.3f},{:+.3f}) yaw {:+.1f}deg -> turn {:+.1f}deg, "
                "dx {:+.3f} m, dy {:+.3f} m (ref {:.2f},{:+.2f})",
                x, y, yaw, turn, dx, dy, ref[0], ref[1])
    return moved_dy


class ChassisNav:
    """--auto-move chassis state, LEARNED within a run (in-memory; run() logs
    the final values for a manual config update).

    Per-DIRECTION leg distances: the arrival correction after each leg feeds
    the distance of the leg that JUST RAN (left = target->source, right =
    source->target). A direction-dependent open-loop travel gain (one shared
    distance converges to a compromise, so one direction lands long every
    visit and gets walked back — the observed overshoot-then-return at the
    source) calibrates out per direction instead. Both distances are clamped
    to CHASSIS_AUTO_STRAFE_DIST_M +/- CHASSIS_LEG_LEARN_CLAMP_M.

    `cur_ref` tracks the case-center y the chassis is currently aligned to.
    Centering refs differ per ITEM (each item's grab point goes to the center
    line), so an item-to-item re-alignment contains a DELIBERATE component
    (ref_change) that must not teach the legs — only the arrival residual
    does. Turns and manual (keyboard) corrections are not tracked."""

    def __init__(self) -> None:
        self.dist_left = float(cfg.CHASSIS_AUTO_STRAFE_DIST_M)
        self.dist_right = float(cfg.CHASSIS_AUTO_STRAFE_DIST_M)
        self.cur_ref = float(cfg.CHASSIS_CENTER_CASE_Y_M)
        # Set when the arrival about to be measured is UNATTRIBUTABLE to the
        # leg that just ran (divert bin-align's open-loop return, or a blind
        # default place — no target centering measured/corrected the right
        # leg): the next centering corrects position but must NOT teach the
        # legs, else e.g. the bin-align return error re-enters dist_left every
        # divert loop and the source stop marches sideways (observed).
        self.skip_next_learn = False

    def dist(self, direction: str) -> float:
        return self.dist_left if direction == "left" else self.dist_right

    def ref_change(self, new_ref: float) -> float:
        """Deliberate strafe component of centering to `new_ref` (old - new,
        +left); updates the tracked ref. 0.0 when the ref is unchanged."""
        d = self.cur_ref - float(new_ref)
        self.cur_ref = float(new_ref)
        return d

    def learn(self, dy: float, at: str) -> None:
        """Feed one centering/adjust RESIDUAL (m, +left; deliberate ref moves
        already removed) at `at` ("source" / "target") into the distance of
        the leg that just ran."""
        if dy == 0.0:
            return
        base = float(cfg.CHASSIS_AUTO_STRAFE_DIST_M)
        lo, hi = base - cfg.CHASSIS_LEG_LEARN_CLAMP_M, base + cfg.CHASSIS_LEG_LEARN_CLAMP_M
        if at == "source":   # correction after the LEFT leg: landed long -> shorten
            new = float(np.clip(self.dist_left + dy, lo, hi))
            logger.info("left-leg distance: {:.3f} -> {:.3f} m (source residual {:+.3f})",
                        self.dist_left, new, dy)
            self.dist_left = new
        else:                # correction after the RIGHT leg: had to go further -> lengthen
            new = float(np.clip(self.dist_right - dy, lo, hi))
            logger.info("right-leg distance: {:.3f} -> {:.3f} m (target residual {:+.3f})",
                        self.dist_right, new, dy)
            self.dist_right = new


def _center_case(bot, layers: int, label: str, station: str,
                 nav: "ChassisNav | None", y_ref: float,
                 plane_z: "float | None" = None,
                 tol_m: "float | None" = None):
    """Navigation-based case centering (--auto-move): detect the case, then
    turn in place so its yaw reads 0 deg (same [-90,90) wrap as
    _center_from_det; deadband/clamp = CHASSIS_ADJUST_MIN/MAX_TURN_DEG) and
    strafe so its center sits at `y_ref` (per ITEM: the item's grab/seat point
    lands on the center line) — BEFORE the pick/place pose is computed. BEV
    detection is most accurate, the reach window widest (yaw 0 = the taught
    wrist branch), and the source/target biases most symmetric, with the item
    point square and dead ahead. The strafe is computed in the post-turn frame
    (a turn swings the ~0.9 m-away case sideways). Up to
    CHASSIS_CENTER_MAX_MOVES correction rounds, each followed by a re-detect.
    Strafes feed the learned per-direction leg distance MINUS the deliberate
    item-to-item ref change (nav.ref_change) so re-alignments don't teach the
    legs; turns are not tracked. Returns the LAST detection (a None /
    not-found detection returns immediately for the caller's normal handling;
    already-centered costs exactly one detect). `tol_m` overrides
    cfg.CHASSIS_CENTER_TOL_M for callers that want a tighter/looser deadband."""
    from .move_chassis import strafe_left, strafe_right, turn_ccw, turn_cw
    skip_learn = False
    if nav is not None:
        skip_learn = nav.skip_next_learn
        nav.skip_next_learn = False
        if skip_learn:
            logger.info("[{}] arrival at {} unattributable (divert/blind place) — "
                        "correcting position without teaching the legs", label, station)
    tol = cfg.CHASSIS_CENTER_TOL_M if tol_m is None else tol_m
    deliberate = nav.ref_change(y_ref) if nav is not None else 0.0
    det = detect(bot, layers, plane_z)
    for _ in range(int(cfg.CHASSIS_CENTER_MAX_MOVES)):
        if det is None or not det.found:
            return det
        if abs(det.base_xy[1] - y_ref) > cfg.CHASSIS_DETECT_Y_GATE_M:
            # The stations are one leg apart and the detector returns the
            # highest-conf OBB anywhere in frame — a far-off "case" is the
            # OTHER station's stack, and centering on it drags the robot away.
            logger.warning("[{}] detection at {} rejected: case y {:+.3f} vs expected "
                           "{:+.3f} (> {:.2f} m gate — other station's stack?)",
                           label, station, det.base_xy[1], y_ref,
                           cfg.CHASSIS_DETECT_Y_GATE_M)
            return None
        yaw = det.base_yaw_deg
        if yaw >= 90.0:                 # long-axis 180-deg wrap, as _center_from_det
            yaw -= 180.0
        turn = 0.0
        if abs(yaw) >= cfg.CHASSIS_ADJUST_MIN_TURN_DEG:
            turn = float(np.clip(yaw, -cfg.CHASSIS_ADJUST_MAX_TURN_DEG,
                                 cfg.CHASSIS_ADJUST_MAX_TURN_DEG))
        # case y in the post-turn base frame (frame rotated CCW by `turn`)
        th = float(np.deg2rad(turn))
        x, y = det.base_xy
        dy = float(-np.sin(th) * x + np.cos(th) * y) - y_ref
        if turn == 0.0 and abs(dy) <= tol:
            return det
        if turn != 0.0:
            logger.info("[{}] centering case at {}: turn {} {:.1f} deg (case yaw {:+.1f})",
                        label, station, "ccw" if turn > 0 else "cw", abs(turn), yaw)
            (turn_ccw if turn > 0 else turn_cw)(bot, angle_deg=abs(turn))
        if abs(dy) > tol:
            dy = float(np.clip(dy, -cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M,
                               cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M))
            logger.info("[{}] centering case at {}: strafe {} {:.3f} m (ref y {:+.3f})",
                        label, station, "left" if dy > 0 else "right", abs(dy), y_ref)
            (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy))
            if nav is not None and not skip_learn:
                nav.learn(dy - deliberate, station)
        deliberate = 0.0   # only the FIRST correction contains the ref change
        det = detect(bot, layers, plane_z)
    return det


def descent_reachable(mover: SuctionMover, pose) -> bool:
    """Pre-flight: solve IK for the WHOLE descent column at the pose's xy —
    from the current EE height down to DESCENT_CHECK_BOTTOM_EE_Z (box floor +
    suction length), regardless of the expected layer — before moving at all.
    Warm-chained downward so successive solves stay on one branch. False (with
    the failing z logged) if any step misses REACH_TOL / limits / collision."""
    x, y = float(pose[0]), float(pose[1])
    rpy = tuple(pose[3:6])
    z0 = float(mover.current_ee_pose()[0][2])
    bottom = float(cfg.DESCENT_CHECK_BOTTOM_EE_Z)
    zs = np.arange(max(z0, bottom), bottom - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M))
    if zs[-1] > bottom + 1e-9:
        zs = np.append(zs, bottom)
    seed = None
    for z in zs:
        sol = mover.solve_pose((x, y, float(z)), rpy,
                               seed=seed, min_motion=seed is not None)
        ok = (sol.pos_err_m <= cfg.REACH_TOL_M) and sol.in_limits and not sol.in_collision
        if not ok:
            logger.error("descent pre-check FAILED at z={:.3f} (err={:.1f}mm, in_lim={}, col={}) "
                         "for xy=({:.3f},{:+.3f})", z, sol.pos_err_m * 1000,
                         sol.in_limits, sol.in_collision, x, y)
            return False
        seed = sol.q
    logger.info("descent pre-check OK: xy=({:.3f},{:+.3f}) z {:.3f} -> {:.3f}", x, y, z0, bottom)
    return True


def _view_park(mover: SuctionMover, label: str) -> None:
    """Move the arm out of the head-camera view: Cartesian park (keeps the
    current EE orientation — safe with or without a held item, tunable via
    config), joint park fallback if unset/unreachable.

    A below-transport start (the lift_to_clear picks/places stop at the
    wall-clear height) first rises STRAIGHT to SAFE_TRANSPORT_Z — the joint
    move to the park otherwise splits the rise across the whole path and
    sweeps sideways while still low (observed: headed left at ~0.95).
    Already-at-transport starts skip it (no-op)."""
    pos, rpy = mover.current_ee_pose()
    if pos[2] < cfg.SAFE_TRANSPORT_Z - 0.01:
        mover.move_ee_vertical(cfg.SAFE_TRANSPORT_Z, rpy)
    parked = False
    if cfg.ARM_VIEW_PARK_EE_POS is not None:
        _, rpy = mover.current_ee_pose()
        parked = mover.move_ee(tuple(cfg.ARM_VIEW_PARK_EE_POS), tuple(rpy)) is not None
        if not parked:
            logger.warning("[{}] Cartesian view-park unreachable — joint park fallback", label)
    if not parked:
        mover.move_joints(np.asarray(cfg.ARM_VIEW_PARK_JOINTS, dtype=np.float64))


def _park_during_legs(label: str, arm_moves, chassis_legs) -> None:
    """Run `arm_moves` (view park / joint home callables) in background threads
    WHILE `chassis_legs()` (one or more chassis moves) runs in the main thread.
    Arm targets are base-frame, so a moving base only drags the world-frame EE
    path sideways — the arms are already lifted clear (transport lift / place
    retreat) when this runs. Joined after the chassis stops; a move that RAISED
    is re-run once synchronously (base still) so the next head-camera detection
    never starts with an arm across the view."""
    failed: list = []
    def _bg(fn):
        try:
            fn()
        except Exception as e:
            logger.warning("[{}] parallel arm move failed ({}) — retrying after "
                           "the chassis stops", label, e)
            failed.append(fn)
    threads = [threading.Thread(target=_bg, args=(fn,), daemon=True)
               for fn in arm_moves]
    for t in threads:
        t.start()
    try:
        chassis_legs()
    finally:
        for t in threads:
            t.join()
        for fn in failed:
            fn()


def _arms_home(bot, mover: SuctionMover) -> None:
    """Safe-home BOTH arms before a failure strafe (lift-if-low first, so a low
    EE doesn't sweep through the box walls on the way home)."""
    logger.info("failure recovery: both arms -> safe home")
    from .go_home import both_arms_home
    both_arms_home(bot, left=mover)


def _detect_bin_xy(bot) -> "tuple[float, float] | None":
    """One bin detection (head camera, full frame) at the current chassis
    position, projected to base_link at DIVERT_BIN_PLANE_Z_M."""
    import detect_bin as dbn  # case_detection sibling (path set at module import)
    rgb = _head_rgb(bot)
    if rgb is None:
        return None
    xy = dbn.find_bin_base_xy(rgb, *_joints(bot), plane_z=cfg.DIVERT_BIN_PLANE_Z_M)
    return None if xy is None else (float(xy[0]), float(xy[1]))


# Last measured bin-align gain (projected-y change / commanded strafe). It
# conflates the projection scale (DIVERT_BIN_PLANE_Z_M vs the real bin plane)
# with the open-loop chassis travel gain — both roughly constant within a run,
# so one call's measurement seeds the next call's first move.
_bin_align_gain = [1.0]


def _align_to_bin(bot, label: str, target_y: "float | None" = None,
                  fallback_right_m: "float | None" = None,
                  max_err_m: "float | None" = None) -> float:
    """Strafe so the detected bin sits at `target_y` (+left, default
    cfg.DIVERT_BIN_TARGET_Y_M) in base_link: detect (head camera, full frame,
    detect_bin) -> strafe the Y error -> re-detect -> ONE more correction if
    the residual exceeds DIVERT_BIN_TOL_M. No bin on the first detect falls
    back to a fixed `fallback_right_m` strafe (default DIVERT_EXTRA_RIGHT_M,
    the original open-loop divert behavior; pass 0.0 to stay put instead —
    used by the seed place, which then places blind as before). `max_err_m`
    optionally rejects a detection whose error exceeds it (treated as no bin
    — the seed place uses this so the SOURCE-side bin, one leg away and in
    view, can't hijack the alignment).

    Moves divide by the OBSERVED gain (measured-y change / commanded move —
    ~1 when DIVERT_BIN_PLANE_Z_M matches the real bin plane and the open-loop
    strafe travels true): a wrong gain made the first move overshoot and the
    raw second correction bounce the chassis straight back to where it started
    (observed live). A gain measured on a re-detect is REMEMBERED
    (_bin_align_gain) and seeds the first move of this and every later call —
    with the per-call gain=1 reset the overshoot-then-return repeated on every
    visit (observed live). A gain far from 1 logs a tune-the-plane warning.

    Returns the NET leftward chassis move in m (negative = net right) so the
    caller can strafe it back after the divert."""
    from .move_chassis import strafe_left, strafe_right
    import detect_bin as dbn  # case_detection sibling (path set at module import)
    if target_y is None:
        target_y = cfg.DIVERT_BIN_TARGET_Y_M
    if fallback_right_m is None:
        fallback_right_m = cfg.DIVERT_EXTRA_RIGHT_M
    net = 0.0
    prev_y: "float | None" = None   # measured bin y before the previous move
    prev_move = 0.0                 # previous commanded strafe (+left)
    for attempt in (1, 2, 3, 4, 5):
        rgb = _head_rgb(bot)
        xy = None if rgb is None else dbn.find_bin_base_xy(
            rgb, *_joints(bot), plane_z=cfg.DIVERT_BIN_PLANE_Z_M)
        if xy is None:
            if attempt == 1:
                if fallback_right_m > 0.0:
                    logger.warning("[{}] no bin detected — fixed {:.2f} m right fallback",
                                   label, fallback_right_m)
                    strafe_right(bot, distance_m=fallback_right_m)
                    return -fallback_right_m
                logger.warning("[{}] no bin detected — staying put", label)
                return 0.0
            logger.warning("[{}] bin lost on re-detect — keeping current position", label)
            return net
        err = xy[1] - target_y   # +: bin left of target -> strafe left
        if max_err_m is not None and abs(err) > max_err_m:
            logger.warning("[{}] bin detection rejected: y {:+.3f} vs target {:+.3f} "
                           "(> {:.2f} m gate — other station's bin?)",
                           label, xy[1], target_y, max_err_m)
            if attempt == 1 and fallback_right_m > 0.0:
                strafe_right(bot, distance_m=fallback_right_m)
                return -fallback_right_m
            return net
        gain = _bin_align_gain[0]
        if prev_y is not None and abs(prev_move) > 1e-6:
            g = (prev_y - xy[1]) / prev_move
            if abs(g - 1.0) > 0.3:
                logger.warning("[{}] bin-align projection gain {:.2f} (expected ~1) — "
                               "tune DIVERT_BIN_PLANE_Z_M", label, g)
            if 0.2 <= g <= 5.0:
                gain = g
                _bin_align_gain[0] = g
        logger.info("[{}] bin @ base xy=({:.3f},{:+.3f}) err {:+.3f} m gain {:.2f} (align {}/2)",
                    label, xy[0], xy[1], err, gain, attempt)
        if abs(err) <= cfg.DIVERT_BIN_TOL_M:
            return net
        move = float(np.clip(err / gain, -cfg.DIVERT_BIN_MAX_STRAFE_M,
                             cfg.DIVERT_BIN_MAX_STRAFE_M))
        (strafe_left if move > 0 else strafe_right)(bot, distance_m=abs(move))
        prev_y, prev_move = float(xy[1]), move
        net += move
    return net


def _divert_sync(mover: SuctionMover, gripper: GripperMover, label: str) -> "bool | None":
    """Two-arm handoff of the suction-held battery to the right-arm gripper,
    run at the TARGET (right) chassis position, SYNCHRONOUS through the place
    sequence — the chassis moves between stations, so the background
    place+home thread sequence.py uses would be dragged along by the next
    strafe. The right-arm HOME is NOT done here: on True the CALLER homes the
    right arm (in parallel with the left-arm view park AND the return strafes,
    joined before the next detection — _park_during_legs). Geometry is sequence.py's _divert (HANDOFF_HOVER_XY /
    HANDOFF_GRIP_OFFSET / PLACE_LOWER_RIGHT_EE_SEQ).

    Returns True once the battery has left the cup (placed, or dropped with the
    gripper empty after a failed place — logged; right arm NOT yet homed);
    False with the battery STILL ON THE CUP if the divert can't start (sequence
    not taught / gripper gripped nothing) so the caller can fall back to the
    case place; None if the place sequence failed with the battery still
    clamped — the right arm is left where it stalled and the run must stop."""
    if not cfg.PLACE_LOWER_RIGHT_EE_SEQ:
        logger.warning("[{}] PLACE_LOWER_RIGHT_EE_SEQ not taught — skipping divert", label)
        return False
    pos, rpy = mover.current_ee_pose()
    tx, ty = cfg.HANDOFF_HOVER_XY
    logger.info("[{}] divert: suction arm -> fixed hover", label)
    if mover.move_ee([tx, ty, float(pos[2])], rpy) is None:
        logger.warning("[{}] handoff hover unreachable — gripping at current pose instead", label)
    pos, _ = mover.current_ee_pose()
    off = cfg.HANDOFF_GRIP_OFFSET
    grasp_pos = [float(pos[i]) + off[i] for i in range(3)]
    logger.info("[{}] divert: right-arm side-grip at suction EE + offset", label)
    if not gripper.grip_at(grasp_pos):
        logger.warning("[{}] gripper reported no object — aborting divert", label)
        return False
    suction_io.release()        # suction lets go; the gripper now holds the battery
    if not gripper.place_ee_seq():
        if gripper.gripper.is_object_grasped():
            logger.error("[{}] right-arm place sequence failed — still holding the "
                         "battery, NOT homing (left where it stalled)", label)
            return None
        logger.error("[{}] right-arm place sequence failed — gripper empty, "
                     "homing (caller)", label)
    return True


def run_item(bot, mover: SuctionMover, label: str, pose_key: str,
             src_layers: int, tgt_layers: int, gripper: "GripperMover | None" = None,
             scan: bool = False, auto: bool = False,
             leg: "ChassisNav | None" = None,
             zt: "ZTracker | None" = None, admittance: bool = False) -> bool:
    """One item's full left->pick->right->place->left cycle.

    `src_layers` / `tgt_layers` are the CURRENT stack heights for this layer
    (the layer loop in run() steps them; cfg.SRC/TGT_LAYERS_REMAINING are only
    the starting values). With `scan`, battery picks read the barcode during
    the descent; a TARGET_BARCODES match diverts to `gripper` at the target
    position instead of the case place. With `auto` (--auto-move), chassis
    legs are automatic and a failed reach pre-check auto-adjusts from the
    detection (up to CHASSIS_ADJUST_MAX_ATTEMPTS) before falling back to the
    interactive keyboard prompt. `admittance` (--admittance) makes the target
    place comply x/y/yaw to contact force while descending, instead of holding
    xy fixed and only reacting via _misseat_recover after a full contact —
    see suction.place(admittance=)."""
    logger.info("=== item: {} ({}) src_layers={} tgt_layers={} ===",
                label, pose_key, src_layers, tgt_layers)
    # Per-item centering ref: put THIS item's grab/seat point (its case-local
    # y offset at yaw 0) on the center line, not the case center.
    y_ref = cfg.CHASSIS_CENTER_CASE_Y_M - resolve_poses((0.0, 0.0, 0.0, 0.0))[pose_key][1]

    # --- SOURCE (already here from the initial move / previous item's return):
    #     detect, pick ---
    src_plane = zt.plane_z("source", src_layers) if zt is not None else None
    if src_plane is not None:
        logger.info("[{}] source warp plane: measured {:.4f} (model {:.4f}, {:+.1f}mm)",
                    label, src_plane, dcb.bev.top_face_z(src_layers),
                    (src_plane - dcb.bev.top_face_z(src_layers)) * 1000.0)
        zt.log_event("plane", "source", label, src_layers,
                     src_plane, dcb.bev.top_face_z(src_layers))
    adjusts = 0
    while True:
        det = (_center_case(bot, src_layers, label, "source", leg, y_ref, src_plane)
               if auto else detect(bot, src_layers, src_plane))
        if det is None or not det.found:
            logger.error("[{}] source detect failed — arms home, then strafe right", label)
            _arms_home(bot, mover)
            strafe(bot, "right", auto, leg)   # don't leave it stuck on the source side
            return False
        det = _refine_det(bot, src_layers, det, src_plane)   # median-of-N for the pose
        _log_det(zt, "source", label, src_layers, det)
        logger.info("[{}] source case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg conf={:.2f}",
                    label, det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.conf)
        center = _center_from_det(det)
        pick_pose = resolve_poses(center)[pose_key]
        if descent_reachable(mover, pick_pose):
            break
        if auto and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
            adjusts += 1
            logger.warning("[{}] pick pose out of reach — auto-adjust {}/{}",
                           label, adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
            dy = _auto_adjust(bot, det, (cfg.SOURCE_CASE_CENTER[0], y_ref))
            if leg is not None:
                leg.learn(dy, "source")
            continue
        # Out of reach: let the operator reposition the chassis, then re-detect
        # (the base frame moved, so the pose must be recomputed) and retry.
        logger.warning("[{}] pick pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to give up", label)
        if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
            logger.error("[{}] giving up — arms home, then strafe right", label)
            _arms_home(bot, mover)
            strafe(bot, "right", auto, leg)
            return False
    _dual_plane_probe(bot, src_layers, src_plane, "source", label, zt)
    pz = zt.predict_ee_z("source", label, src_layers) if zt is not None else None
    exp_z = pz if pz is not None else det.top_face_z + cfg.SUCTION_LENGTH_M
    # lift_to_clear: the pick lift stops at the wall-clear height and the
    # strafe right below starts immediately; the parallel view park (carry
    # pose, z = SAFE_TRANSPORT_Z) finishes the rise. User-verified 0806: the
    # held part's bottom clears the box walls at that height.
    if scan and label.startswith("battery"):
        # barcode-gated pick: scan during the descent (sweep bounded by the
        # DETECTED case center, not the taught one)
        res = mover.pick_gated(pick_pose, case_center=center, expected_z=exp_z,
                               lift_to_clear=True)
    else:
        res = mover.pick(pick_pose, expected_z=exp_z, lift_to_clear=True)
    if not res.success:
        logger.error("[{}] pick failed: {} — arms home, then strafe right", label, res.reason)
        if zt is not None:
            zt.log_event("pick_" + res.reason, "source", label, src_layers,
                         res.contact_ee_z, exp_z)
        _arms_home(bot, mover)
        strafe(bot, "right", auto, leg)   # not on the target side yet -> move off the source side
        return False
    if zt is not None:
        zt.record("source", label, src_layers, res.contact_ee_z)
    if res.barcode is not None:
        logger.info("[{}] barcode {!r} (target={})", label, res.barcode, is_target(res.barcode))

    # --- TARGET: park the arm (clear the head view) IN PARALLEL with the
    #     strafe right; a barcode-matched battery diverts to the right-arm
    #     gripper, everything else detects + places in the case ---
    _park_during_legs(label, [lambda: _view_park(mover, label)],
                      lambda: strafe(bot, "right", auto, leg))
    if gripper is not None and is_target(res.barcode):
        from .move_chassis import strafe_left, strafe_right
        logger.info("[{}] TARGET battery {!r} — diverting to right-arm gripper",
                    label, res.barcode)
        # position the (base_link-fixed) right-arm place over the divert bin:
        # strafe so the detected bin sits at DIVERT_BIN_TARGET_Y_M; the net
        # move is strafed back before anything that needs the target geometry
        back = -_align_to_bin(bot, label)   # +: strafe left to return
        diverted = _divert_sync(mover, gripper, label)
        if diverted is None:
            # battery off the cup but the right arm stalled holding it — stop here
            logger.error("[{}] stopping on the right side (right arm holds the battery)", label)
            return False
        if leg is not None:
            # bin-align + its return are open-loop and the right-leg arrival was
            # never measured — the NEXT centering must not teach the legs
            leg.skip_next_learn = True
        if diverted:
            logger.info("[{}] right arm -> home + left view park (parallel "
                        "with the return strafes)", label)
            def _return_legs():
                if abs(back) > 1e-3:
                    (strafe_left if back > 0 else strafe_right)(bot, distance_m=abs(back))
                strafe(bot, "left", auto, leg)
            _park_during_legs(label,
                              [lambda: gripper.move_joints(gripper._home_seed),
                               lambda: _view_park(mover, label)],
                              _return_legs)
            return True
        logger.warning("[{}] divert failed — seating in the case instead", label)
        if abs(back) > 1e-3:
            (strafe_left if back > 0 else strafe_right)(bot, distance_m=abs(back))
    adjusts = 0
    redetects = 0
    tgt_plane = zt.plane_z("target", tgt_layers) if zt is not None else None
    if tgt_plane is not None:
        logger.info("[{}] target warp plane: measured {:.4f} (model {:.4f}, {:+.1f}mm)",
                    label, tgt_plane, dcb.bev.top_face_z(tgt_layers),
                    (tgt_plane - dcb.bev.top_face_z(tgt_layers)) * 1000.0)
        zt.log_event("plane", "target", label, tgt_layers,
                     tgt_plane, dcb.bev.top_face_z(tgt_layers))
    pz, mtol = (zt.place_expectation("target", label, tgt_layers)
                if zt is not None else (None, None))
    # The run's FIRST case: the target is empty BY DEFINITION, so case
    # detection (and its centering moves) is skipped entirely — the only case
    # in view would be the SOURCE stack one leg away (observed: it won the
    # detection and dragged the robot back left). Later layers, the layer's
    # batteries, and resumed runs (TGT_LAYERS_REMAINING > 1) always detect.
    seed = label == "case" and tgt_layers <= 1
    while True:
        tdet = None
        if not seed:
            tdet = (_center_case(bot, tgt_layers, label, "target", leg, y_ref, tgt_plane)
                    if auto else detect(bot, tgt_layers, tgt_plane))

        if seed:
            logger.info("[{}] empty target (first case) -> seed place, no case "
                        "detection", label)
            seed_xy = tuple(cfg.TARGET_DEFAULT_CASE_CENTER[:2])
            if auto:
                # The default pose is base_link-fixed, so without this the
                # first case inherits the full leg arrival error and the whole
                # stack builds off it. Strafe so the BIN center sits where the
                # default pose will land the case center (TARGET_DEFAULT y);
                # no bin detected -> blind, as before.
                _align_to_bin(bot, label,
                              target_y=cfg.TARGET_DEFAULT_CASE_CENTER[1],
                              fallback_right_m=0.0,
                              max_err_m=cfg.CHASSIS_DETECT_Y_GATE_M)
                # CLOSED-LOOP seed: one more bin detection AFTER the align, and
                # the place center comes from it (+ the measured bbox bias,
                # SEED_BIN_CENTER_OFFSET) — the strafe's accepted residual
                # (DIVERT_BIN_TOL_M) and the leg's x error then never reach the
                # placement; the ARM absorbs them (0806: occasional wall-catch,
                # detected x ~47mm forward of true). Detect fail -> default.
                bxy = _detect_bin_xy(bot)
                if (bxy is not None and abs(bxy[1] - cfg.TARGET_DEFAULT_CASE_CENTER[1])
                        <= cfg.CHASSIS_DETECT_Y_GATE_M):
                    seed_xy = (bxy[0] + cfg.SEED_BIN_CENTER_OFFSET[0],
                               bxy[1] + cfg.SEED_BIN_CENTER_OFFSET[1])
                    logger.info("[{}] seed place from the detected bin: center "
                                "({:.3f},{:+.3f}) -> place ({:.3f},{:+.3f})", label,
                                bxy[0], bxy[1], seed_xy[0], seed_xy[1])
                else:
                    logger.warning("[{}] bin re-detect failed/rejected — seed at the "
                                   "default pose", label)
            # Seed at the SOURCE's detected yaw (carried from this item's pick,
            # `center`) instead of base-frame 0: the wrist then does NO
            # de-rotation in transit and the stack mirrors the source stack's
            # orientation rather than the chassis heading at arrival.
            seed_center = (*seed_xy, cfg.TARGET_DEFAULT_CASE_CENTER[2], center[3])
            logger.info("[{}] seed place at the source yaw {:+.1f} deg", label,
                        float(np.rad2deg(center[3])))
            place_pose = resolve_poses(seed_center)[pose_key]
            intended = [seed_center[0], seed_center[1], float(np.rad2deg(center[3]))]
            exp_z = None
            # misseat gate off the TAUGHT seat z (exp_z stays None -> place()
            # falls back to the pose z): a rim/wall landing contacts several cm
            # above the bin floor and is HELD instead of released (0806).
            mtol = cfg.SEED_MISSEAT_TOL_M
            if leg is not None:
                # right-leg arrival never measured against the CASE reference
                # (bin-anchored / blind place) — the next centering corrects
                # position without teaching the legs
                leg.skip_next_learn = True

        elif tdet is not None and tdet.found:
            logger.info("[{}] target case found -> aligned place", label)
            tdet = _refine_det(bot, tgt_layers, tdet, tgt_plane)   # median-of-N for the pose
            _log_det(zt, "target", label, tgt_layers, tdet)
            tc = _center_from_det(tdet)
            place_pose = resolve_poses(tc)[pose_key]
            intended = [tc[0], tc[1], float(np.rad2deg(tc[3]))]
            exp_z = pz if pz is not None else tdet.top_face_z + cfg.SUCTION_LENGTH_M

        else:
            # A case MUST already be at the target (seeded/built) — a miss here
            # is a DETECTION failure, not an empty target, and blind-stacking
            # at the base-fixed default pose would land misaligned on the real
            # stack. Re-detect once, then hand it to the operator.
            if redetects < 1:
                redetects += 1
                logger.warning("[{}] target case NOT detected (stack expected) — "
                               "re-detecting once", label)
                continue
            logger.warning("[{}] target case NOT detected (stack expected) — adjust the "
                           "chassis (f/b/l/r), `d` to re-detect + retry, `q` to stop "
                           "(item still held)", label)
            if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping on the right side (item still held)", label)
                return False
            continue
        if cfg.PLACE_X_LAYER_TRIM_M and tgt_layers > 1:
            # layer-height-dependent forward bias compensation (see config)
            trim = cfg.PLACE_X_LAYER_TRIM_M * (tgt_layers - 1)
            place_pose = (place_pose[0] + trim, *place_pose[1:])
            intended[0] += trim
            logger.info("[{}] place x trim {:+.1f} mm (tgt_layers={})",
                        label, trim * 1000, tgt_layers)
        if cfg.PLACE_X_PLANE_TRIM_M and tgt_plane is not None:
            # measured-plane xy shift compensation (see config: the taught
            # offsets were tuned against the MODEL plane's constant bias)
            place_pose = (place_pose[0] + cfg.PLACE_X_PLANE_TRIM_M, *place_pose[1:])
            intended[0] += cfg.PLACE_X_PLANE_TRIM_M
            logger.info("[{}] place x plane trim {:+.1f} mm (measured warp plane)",
                        label, cfg.PLACE_X_PLANE_TRIM_M * 1000)
        if cfg.PLACE_YAW_TRIM_RAD and label.startswith("battery"):
            # systematic in-hand twist compensation — batteries only: the
            # estimate came from battery seats, and a wrong trim on the CASE
            # place moves its corners ~6mm/1.8deg against the 1-2mm jig fit
            place_pose = (*place_pose[:5], place_pose[5] + cfg.PLACE_YAW_TRIM_RAD)
            logger.info("[{}] place yaw trim {:+.1f} deg", label,
                        float(np.rad2deg(cfg.PLACE_YAW_TRIM_RAD)))
        if descent_reachable(mover, place_pose):
            break
        # auto-adjust needs a detection to steer by — the no-case default pose
        # is fixed in base_link, so it goes straight to the keyboard prompt
        if auto and tdet is not None and tdet.found and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
            adjusts += 1
            logger.warning("[{}] place pose out of reach — auto-adjust {}/{}",
                           label, adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
            dy = _auto_adjust(bot, tdet, (cfg.TARGET_DEFAULT_CASE_CENTER[0], y_ref))
            if leg is not None:
                leg.learn(dy, "target")
            continue
        # holding the item — no auto-recovery; reposition + retry, or stop here
        logger.warning("[{}] place pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to stop (item still held)", label)
        if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
            logger.error("[{}] stopping on the right side (item still held)", label)
            return False
    _dual_plane_probe(bot, tgt_layers, tgt_plane, "target", label, zt)
    # misseat check only with a measured-anchored expectation (own / sibling /
    # case anchor, each with its tolerance) — the model plane has been seen off
    # by more than any of those tolerances (0804 layer 5)
    # lift_to_clear: place stops the lift at the wall-clear height (0.95, cup
    # empty) — the return strafe below starts right away and the view park
    # (target z = SAFE_TRANSPORT_Z) finishes the rise in parallel
    pres = mover.place(place_pose, expected_z=exp_z, misseat_tol_m=mtol,
                       lift_to_clear=True,
                       admittance=admittance and label.startswith("battery"),
                       case_search=admittance and label == "case")
    if zt is not None and pres is not None:
        # contact diagnostics: tared base wrench + cmd-vs-measured EE yaw at
        # the (final) contact, and the recovery outcome if any retries ran
        ci = getattr(pres, "contact_info", None) or {}
        if "fz" in ci:
            zt.log_event("contact_wrench", "target", label, tgt_layers, ci["fz"],
                         resid="fx=%+.1f;fy=%+.1f;mx=%+.2f;my=%+.2f;mz=%+.3f"
                               % (ci.get("fx", 0.0), ci.get("fy", 0.0),
                                  ci.get("mx", 0.0), ci.get("my", 0.0), ci.get("mz", 0.0)))
        if "yaw_track_err_deg" in ci:
            zt.log_event("contact_yaw", "target", label, tgt_layers,
                         ci["yaw_meas_deg"], ci["yaw_cmd_deg"],
                         resid="%+.2fdeg" % ci["yaw_track_err_deg"])
        for h in (getattr(pres, "recover_history", None) or []):
            zt.log_event(
                "recover_step", "target", label, tgt_layers,
                resid="a%d;%s;dyaw=%+.1fdeg;dxy=(%+.1f/%+.1f)mm;%s;z=%s;fx=%s;fy=%s;mz=%s" % (
                    h["attempt"], h["mode"], h["dyaw_deg"], h["dx_mm"], h["dy_mm"],
                    h["reason"],
                    "n/a" if h["z_mm"] is None else "%+.1fmm" % h["z_mm"],
                    "n/a" if h["fx"] is None else "%+.1f" % h["fx"],
                    "n/a" if h["fy"] is None else "%+.1f" % h["fy"],
                    "n/a" if h["mz"] is None else "%+.3f" % h["mz"]))
        if getattr(pres, "recover_attempts", 0):
            zt.log_event("recovered" if pres.success else "recover_fail",
                         "target", label, tgt_layers,
                         resid="attempts=%d" % pres.recover_attempts)
    if pres is not None and not getattr(pres, "success", True):
        if zt is not None:
            zt.log_event("place_" + pres.reason, "target", label, tgt_layers,
                         pres.contact_ee_z, exp_z)
        if pres.reason == "unreachable" and pres.contact_ee_z is None:
            # the HOVER leg failed BEFORE the release gate — the part is still
            # on the cup, so continuing would carry it into the next item
            logger.error("[{}] place failed: {} (part still held) — stopping on "
                         "the right side", label, pres.reason)
            return False
        # every other failure passed the operator release gate (hand-guided
        # seat + Enter -> blow-off): the item is on the target — log it and
        # keep the run moving; the z anchor stays clean (success-gated below)
        logger.warning("[{}] place failed ({}) — operator resolved it at the "
                       "gate, continuing with the next item", label, pres.reason)
    if zt is not None and pres is not None and getattr(pres, "success", True):
        zt.record("target", label, tgt_layers, pres.contact_ee_z)

    if cfg.PLACE_VERIFY_DETECT and label == "case" and zt is not None:
        # --- place verification: park the arm to clear the view (SYNC — the
        #     check needs a still chassis), re-detect the just-placed case,
        #     log landed-vs-intended in the SAME base frame, then strafe ---
        _view_park(mover, label)
        chk = detect(bot, tgt_layers, zt.plane_z("target", tgt_layers))
        if chk is not None and chk.found:
            dyaw = (chk.base_yaw_deg - intended[2] + 90.0) % 180.0 - 90.0
            logger.info("[{}] place check: landed ({:+.1f}, {:+.1f}) mm, {:+.2f} deg "
                        "vs intended", label,
                        (chk.base_xy[0] - intended[0]) * 1000.0,
                        (chk.base_xy[1] - intended[1]) * 1000.0, dyaw)
            zt.log_event("place_chk_x", "target", label, tgt_layers,
                         chk.base_xy[0], intended[0])
            zt.log_event("place_chk_y", "target", label, tgt_layers,
                         chk.base_xy[1], intended[1])
            zt.log_event("place_chk_yaw", "target", label, tgt_layers,
                         chk.base_yaw_deg, intended[2], resid=f"{dyaw:+.2f}deg")
        else:
            logger.warning("[{}] place check: case not detected — no landing sample", label)
        strafe(bot, "left", auto, leg)
    else:
        # --- park again (clear the head view for the SOURCE detect) IN PARALLEL
        #     with the return LEFT ---
        _park_during_legs(label, [lambda: _view_park(mover, label)],
                          lambda: strafe(bot, "left", auto, leg))
    return True


def run(bot, mover: SuctionMover, gripper: "GripperMover | None" = None,
        scan: bool = False, auto: bool = False, admittance: bool = False) -> bool:
    """Layer loop: each layer runs the full item set (case + 2 batteries), then
    the stacks step (source -1, target +1) so the BEV warp plane tracks the
    shrinking source / growing target. Loops until the source is exhausted.
    `gripper`/`scan`/`auto`/`admittance` are passed through to run_item
    (barcode divert / --auto-move / --admittance); with `auto`, arrival
    residuals feed back into learned per-direction leg distances (ChassisNav)
    used by every later strafe leg."""
    # The OPERATOR positions the chassis at the SOURCE station once (keyboard,
    # move_chassis grammar — replaces the old fixed initial left leg); each
    # item returns here at its end, so a failed pick just stops (no
    # strafe-right, robot left where it is). Residual is absorbed by centering.
    leg = ChassisNav() if auto else None
    # measured-contact z anchors (expected z + warp plane per column), with the
    # per-run error CSV (contact-vs-predicted, plane measured-vs-model, failures)
    zt = ZTracker(log_path=None if cfg.ZTRACK_LOG_DIR is None else
                  f"{cfg.ZTRACK_LOG_DIR}/ztrack_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    logger.info("position the chassis at the SOURCE station (case in front), then `d`")
    if not _manual_strafe(bot, "start"):
        logger.error("start positioning aborted (`q`) — run cancelled")
        return False
    src, tgt = cfg.SRC_LAYERS_REMAINING, cfg.TGT_LAYERS_REMAINING
    layer = 0
    try:
        while src >= 1:
            layer += 1
            logger.info("=== layer {}: source stack {}, target stack {} ===", layer, src, tgt)
            for label, key in ITEMS:
                if not run_item(bot, mover, label, key, src, tgt,
                                gripper=gripper, scan=scan, auto=auto, leg=leg, zt=zt,
                                admittance=admittance):
                    logger.error("stopping at layer {} item {} (robot left where it is) — "
                                 "to resume, set SRC_LAYERS_REMAINING={} TGT_LAYERS_REMAINING={}",
                                 layer, label, src, tgt)
                    return False
            src -= 1
            tgt += 1
        logger.info("=== all {} layers moved ===", layer)
        return True
    finally:
        zt.close()
        base = cfg.CHASSIS_AUTO_STRAFE_DIST_M
        if leg is not None and (leg.dist_left != base or leg.dist_right != base):
            logger.info("learned leg distances this run: left {:.3f} m, right {:.3f} m "
                        "(config CHASSIS_AUTO_STRAFE_DIST_M = {:.2f})",
                        leg.dist_left, leg.dist_right, base)


def _main() -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    KNOWN_FLAGS = {"--gripper", "--auto-move", "--dashboard", "--admittance"}
    unknown = [a for a in sys.argv[1:] if a not in KNOWN_FLAGS]
    if unknown:
        raise ValueError(f"unknown flag(s) {unknown} — choose from {sorted(KNOWN_FLAGS)}")

    use_gripper = "--gripper" in sys.argv   # barcode scan + divert (as sequence.py)
    auto_move = "--auto-move" in sys.argv   # automatic chassis legs + auto-adjust
    use_dashboard = "--dashboard" in sys.argv  # spool camera/joints/EE/wrench (as sequence.py)
    use_admittance = "--admittance" in sys.argv  # target place complies to contact force (experimental)

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION + CHASSIS (strafe L/R per item):")
    for label, key in ITEMS:
        logger.warning("   {} ({})", label, key)
    if use_gripper:
        logger.warning("Barcode divert ENABLED (target codes -> right gripper, at the target side).")
    if auto_move:
        logger.warning("AUTO chassis ENABLED: {:.2f} m legs + detection-based adjust.",
                       cfg.CHASSIS_AUTO_STRAFE_DIST_M)
    if use_dashboard:
        logger.warning("Dashboard spool ENABLED — view with run_dashboard_demo.sh.")
    if use_admittance:
        logger.warning("Admittance place ENABLED (experimental, unverified gains) — "
                       "target descent complies x/y/yaw to contact force.")
    logger.warning("Clear the strafe path. E-stop in reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        # Tilt the head down so the box is in view (same as live_bev/capture); the
        # BEV homography uses the live joints, so ~30 deg matches the training data.
        set_head_pitch(bot, angle=15.0)
        publisher = None
        if use_dashboard:
            from .dashboard_publish import DashboardPublisher
            publisher = DashboardPublisher(bot).start()
        try:
            with SuctionMover(bot) as m:
                release = m.software_estop_active()
                if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                    return
                if not m.ensure_ready(release_estop=release):
                    logger.error("arm not ready — aborting")
                    return
                gripper = None
                if use_gripper:
                    gripper = GripperMover(bot)
                    gripper.ensure_ready(release_estop=release)
                    if not gripper.initialize():
                        logger.warning("gripper unavailable — divert disabled")
                        gripper = None
                # Start clean: BOTH arms safe-homed (lift-if-low first, so an arm
                # left down in a box by a previous run doesn't sweep the walls).
                logger.info("-> both arms safe home")
                from .go_home import both_arms_home
                both_arms_home(bot, left=m)
                ok = run(bot, m, gripper=gripper, scan=use_gripper, auto=auto_move,
                        admittance=use_admittance)
                logger.info("-> home")
                m.move_joints(m._home_seed)
                logger.info("sequence {}", "OK" if ok else "FAILED")
        finally:
            if publisher is not None:
                publisher.stop()


if __name__ == "__main__":
    _main()
