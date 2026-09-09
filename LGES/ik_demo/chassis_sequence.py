"""Chassis-based, detection-driven pick & place.

Loops over LAYERS until ONE case is left in the source: each layer runs the
full item set (case -> battery_1 -> battery_2), then the stack heights step
automatically (source -1, target +1) so the BEV warp plane stays on the true
top face. The last case (no batteries) goes to the bin box on its own.
cfg.SRC/TGT_LAYERS_REMAINING are only the STARTING physical heights (defaults
for the values the task menu asks for at start).

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

    --box        after the last case is away and the chassis is back: the RIGHT
                 arm grips the paper box's right wall (BEV OBB detection, rim at
                 cfg.BOX_RIM_Z_M), lifts it BOX_LIFT_TEST_M, sets it back
                 down, RELEASES, retreats straight up and homes. Needs the
                 Robotiq to answer at start-up (else the step is skipped with
                 a warning).
    barcode divert (always on; was --gripper):
                 battery picks scan the barcode during the descent (pick_gated);
                 a TARGET_BARCODES match is carried LEFT from the source
                 (fixed DIVERT_CASE_STRAFE_LEFT_M strafe, open-loop) and
                 suction-placed into the DIVERT CASE there (BEV-detected,
                 aligned like a normal place) instead of being seated in the
                 target case. Slot order is remembered across the run: the
                 first target goes to the LEFT slot (BAT_SRC_2), the second to
                 the RIGHT slot (BAT_SRC_1); at most two are expected. The
                 chassis then strafes back right to the source. A failed
                 detect/reach falls back to the normal target-case place.
    auto chassis (always on; was --auto-move):
                 chassis legs run automatically: fixed CHASSIS_AUTO_STRAFE_DIST_M
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

    python -m LGES.ik_demo.chassis_sequence [--dashboard] [--state-publish]
    python -m LGES.ik_demo.chassis_sequence --lid | --box | --box-lid | --case-bin   # first task without the menu

One robot session runs many tasks: after each task the arms home and the task
menu comes back (`q` there ends the session).

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
from scipy.spatial.transform import Rotation

from . import config as cfg
from .barcode import is_target
from .config import resolve_poses
from .box_pick import run_box_pick
from .gripper import GripperMover
from .arm import ArmMover, connect_robot, move_torso
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


_RUN_STAMP: list[str] = []


def run_stamp() -> str:
    """One timestamp per process, shared by the run log and the ztrack CSV so a
    run's two files sit next to each other under the same name."""
    if not _RUN_STAMP:
        _RUN_STAMP.append(time.strftime("%Y%m%d_%H%M%S"))
    return _RUN_STAMP[0]


def setup_logging() -> str:
    """Install the run's log sinks and return the stamp. Called from _main, so
    it covers EVERY mode — it used to sit inside run(), which meant --lid wrote
    no log file at all and printed the per-tick traces to the terminal (0904)."""
    stamp = run_stamp()
    if not cfg.DESCENT_TRACE_TO_TERMINAL:
        # Keep the per-tick traces OUT of the terminal but IN the run log:
        # replace loguru's default stderr sink (id 0) with a filtered one.
        # Filtering by function is exact here — _track_trace does nothing else,
        # so no operator-facing line is lost (the corner dbg line shares its
        # function with the wall-latch and contact messages, so it is left
        # alone; DESCENT_TRACE_S = 0.0 silences everything if needed).
        try:
            logger.remove(0)
        except ValueError:
            pass                            # already replaced by a caller
        logger.add(sys.stderr, filter=lambda r: r["function"] != "_track_trace")
    if cfg.RUN_LOG_DIR is not None:
        Path(cfg.RUN_LOG_DIR).mkdir(parents=True, exist_ok=True)
        logger.add(f"{cfg.RUN_LOG_DIR}/run_{stamp}.log", level="INFO")
        logger.info("run log -> {}/run_{}.log  (per-tick traces: file only)",
                    cfg.RUN_LOG_DIR, stamp)
    return stamp


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
        # EVERY measured contact, not just the anchor — the chained expectation
        # (expected_ee_z) reads what an item physically rests on as of THIS run
        self._last: dict[tuple[str, str], tuple[float, int]] = {}
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

    def expected_ee_z(self, station: str, label: str,
                      layers: int) -> "float | None":
        """Expected contact ee-z for a DESCENT: the chained prediction if the
        thing this item rests on has been measured at the right layer, else the
        own-anchor extrapolation.

        The chain replaces a nominal-pitch extrapolation with this run's own
        measurements (see cfg.BATTERY_SEAT_ABOVE_CASE_M). Kept separate from
        predict_ee_z on purpose: that one still drives the misalign/anchor
        logic and the warp plane, which are calibrated against the own-anchor
        model and must not start moving with the chain.
        """
        z = self._chained(station, label, layers)
        return self.predict_ee_z(station, label, layers) if z is None else z

    def _chained(self, station: str, label: str,
                 layers: int) -> "float | None":
        is_bat = label.startswith("battery")
        if station == "source":
            # the batteries sit IN the case, and at the source they contact at
            # the same height as it (within 1.6mm) — no offset
            if not is_bat:
                return None
            z, ly = self._last.get(("source", "case"), (None, None))
            return z if (z is not None and ly == layers) else None
        if is_bat:
            # seats in THIS layer's case, one compartment depth up
            z, ly = self._last.get(("target", "case"), (None, None))
            return (z + float(cfg.BATTERY_SEAT_ABOVE_CASE_M)
                    if (z is not None and ly == layers) else None)
        # a case seats on the PREVIOUS layer's batteries — on the HIGHEST of
        # them, which is also the safe (high) choice for a creep line
        top = None
        for lb in ("battery_1", "battery_2"):
            z, ly = self._last.get(("target", lb), (None, None))
            if z is not None and ly == layers - 1:
                top = z if top is None else max(top, z)
        return None if top is None else top + float(cfg.CASE_SEAT_ABOVE_BATTERY_M)

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
        # CHAIN FIRST: what this item rests on, measured this run (see
        # expected_ee_z). It beats the own-anchor extrapolation because the
        # nominal LAYER_PITCH_M drifts, and it also covers a column's FIRST
        # place — the case seat is measured minutes earlier, so a battery no
        # longer has to fall through to the borrow paths below.
        ch = self._chained(station, label, layers)
        if ch is not None:
            return ch, cfg.PLACE_MISSEAT_TOL_M
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
        self._last[(station, label)] = (float(z), int(layers))
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


def _log_pick_depth(bot, label: str, pick_pose, plane_z: "float | None",
                    layers: int, expected_ee_z: float, zt: "ZTracker | None") -> None:
    """Diagnosis only: what the ZED depth reads UNDER THE PICK POINT (the
    case-frame slot offset the cup descends to), before the descent. The
    detection is of the CASE top face — this says whether the depth camera
    actually sees a battery at the slot, and at what height, next to the
    contact expectation the descent uses (CSV row depth_pick). Never raises."""
    try:
        import depth_plane as dp

        rgb = _head_rgb(bot, fresh=False)
        depth = bot.sensors.head_camera.get_depth()
        if rgb is None or depth is None:
            logger.warning("[{}] pick-point depth: no frame (rgb={}, depth={})",
                           label, rgb is not None, depth is not None)
            return
        q_torso, q_head = _joints(bot)
        guess = float(plane_z if plane_z is not None else dcb.bev.top_face_z(layers))
        xy = (float(pick_pose[0]), float(pick_pose[1]))
        z, n_px, spread = dp.plane_from_depth(depth, rgb.shape, q_torso, q_head, xy, guess)
        if z is None:
            logger.warning("[{}] pick-point depth @ ({:.3f},{:+.3f}): only {} valid px "
                           "— battery NOT seen by depth", label, xy[0], xy[1], n_px)
            if zt is not None:
                zt.log_event("depth_pick_miss", "source", label, layers, None,
                             expected_ee_z, resid=f"px={n_px}")
            return
        ee_z = z + cfg.SUCTION_LENGTH_M
        logger.info("[{}] DEPTH @ pick point ({:.3f},{:+.3f}): surface z={:.4f} "
                    "({} px, relief {:.0f}mm) -> ee_z {:.4f} vs expected {:.4f} "
                    "({:+.1f}mm)", label, xy[0], xy[1], z, n_px, spread * 1000.0,
                    ee_z, expected_ee_z, (ee_z - expected_ee_z) * 1000.0)
        if zt is not None:
            zt.log_event("depth_pick", "source", label, layers, ee_z, expected_ee_z)
    except Exception as e:  # noqa: BLE001 — a log line must not stop a run
        logger.warning("[{}] pick-point depth failed ({})", label, e)


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
           leg: "ChassisNav | None" = None, extra_m: float = 0.0) -> bool:
    """One chassis leg toward `direction`: fixed-DISTANCE automatic leg when
    `auto` (--auto-move, overrides CHASSIS_MANUAL; `leg` carries the learned
    distance), manual (interactive) when cfg.CHASSIS_MANUAL, else the fixed
    open-loop speed*time strafe. `extra_m` folds a KNOWN deliberate offset
    (e.g. the next item's centering ref) into this same move, so the
    detection-based centering that follows only has to correct real residual.
    Returns False only when the user gives up a manual leg (`q`)."""
    if auto:
        from .move_chassis import strafe_left, strafe_right
        fn = strafe_left if direction == "left" else strafe_right
        base = leg.dist(direction) if leg is not None else cfg.CHASSIS_AUTO_STRAFE_DIST_M
        fn(bot, distance_m=base + extra_m,
           speed=cfg.CHASSIS_LEG_SPEED_MS)   # long legs only — corrections stay slow
        return True
    if cfg.CHASSIS_MANUAL:
        return _manual_strafe(bot, direction)
    v = cfg.CHASSIS_STRAFE_SPEED_MS if direction == "left" else -cfg.CHASSIS_STRAFE_SPEED_MS
    logger.info("chassis strafe {} ({:.2f} m/s x {:.1f}s)", direction, v, cfg.CHASSIS_STRAFE_TIME_S)
    bot.chassis.move_sideways(v, wait_time=cfg.CHASSIS_STRAFE_TIME_S)
    time.sleep(cfg.CHASSIS_SETTLE_S)
    return True


def _column_quiet(mover: SuctionMover, x: float, y: float, rpy,
                  z_top: float, z_bottom: float) -> bool:
    """descent_reachable's column sweep (warm-chained IK, z_top -> z_bottom in
    DESCENT_CHECK_STEP_M steps) WITHOUT its logging — for a search over dozens
    of candidate chassis offsets, where one line per miss is noise."""
    zs = np.arange(max(z_top, z_bottom), z_bottom - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M))
    if zs[-1] > z_bottom + 1e-9:
        zs = np.append(zs, z_bottom)
    seed = None
    for z in zs:
        sol = mover.solve_pose((x, y, float(z)), rpy, seed=seed, min_motion=seed is not None)
        if not (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision):
            return False
        seed = sol.q
    return True


def _auto_adjust(bot, mover: SuctionMover, det, pose) -> "float | None":
    """One chassis correction for a resolved `pose` (x, y, z, r, p, yaw) that
    failed the reach pre-check: the SMALLEST forward/back + strafe that puts
    the pose's descent column inside reach.

    The old version dragged BOTH axes onto a taught reference point (x from
    SOURCE_CASE_CENTER / TARGET_DEFAULT_CASE_CENTER, y from the center line):
    a pose 4 cm short in x also got strafed 20-30 cm sideways to a y that was
    already fine, and a spot that solved 3 cm away was passed over for the
    reference. The lid place moved to nearest-solvable-spot for the same reason
    (_nearest_place_spot, 0904); this is that for the case/battery columns.

    ``det`` (a case detection, or None for a bin-anchored place) still supplies
    the in-place turn that squares the case yaw up first, as before; the pose is
    then expressed in the post-turn frame. Offsets are searched on a
    CHASSIS_ADJUST_STEP_M grid out to CHASSIS_ADJUST_MAX_TRANSLATE_M (L1),
    nearest-first; a candidate is taken when its column solves AND still solves
    CHASSIS_ADJUST_REACH_MARGIN_M off in +-x / +-y (so the open-loop move does
    not land on the edge of reach — 0906's lid lesson); if nothing has that
    slack the nearest bare pass is used. The caller re-detects and re-checks
    reach after every call.

    Returns the APPLIED lateral move (m, +left; 0.0 under the deadband) so the
    caller can keep feeding the learned leg distances, or None when NO offset
    within the clamp solves — the caller should go to the keyboard instead of
    spending another round on the same answer."""
    from .move_chassis import (move_backward, move_forward, strafe_left,
                               strafe_right, turn_ccw, turn_cw)
    t0 = time.monotonic()
    turn = 0.0
    if det is not None:
        yaw = det.base_yaw_deg
        if yaw >= 90.0:             # long-axis 180-deg wrap, as _center_from_det
            yaw -= 180.0
        if abs(yaw) >= cfg.CHASSIS_ADJUST_MIN_TURN_DEG:
            turn = float(np.clip(yaw, -cfg.CHASSIS_ADJUST_MAX_TURN_DEG,
                                 cfg.CHASSIS_ADJUST_MAX_TURN_DEG))
            (turn_ccw if turn > 0 else turn_cw)(bot, angle_deg=abs(turn))
    # the pose in the post-turn base frame (frame rotated CCW by `turn`)
    th = float(np.deg2rad(turn))
    px, py = float(pose[0]), float(pose[1])
    xr = float(np.cos(th) * px + np.sin(th) * py)
    yr = float(-np.sin(th) * px + np.cos(th) * py)
    rpy = (float(pose[3]), float(pose[4]), float(pose[5]) - th)
    z_top = max(float(mover.current_ee_pose()[0][2]), float(cfg.DESCENT_CHECK_BOTTOM_EE_Z))
    z_bot = float(cfg.DESCENT_CHECK_BOTTOM_EE_Z)
    step = float(cfg.CHASSIS_ADJUST_STEP_M)
    lim = float(cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M)
    m = float(cfg.CHASSIS_ADJUST_REACH_MARGIN_M)
    n = int(round(lim / step))
    # chassis moves (dx forward, dy left) -> the pose moves (-dx, -dy) in base
    cands = sorted(((i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
                    if 0 < abs(i) + abs(j) <= n),
                   key=lambda o: (abs(o[0]) + abs(o[1]), abs(o[0]), abs(o[1])))
    bare = None
    found = None
    for dx, dy in cands:
        x, y = xr - dx, yr - dy
        if not _column_quiet(mover, x, y, rpy, z_top, z_bot):
            continue
        if bare is None:
            bare = (dx, dy)
        if all(_column_quiet(mover, x + ox, y + oy, rpy, z_top, z_bot)
               for ox, oy in ((m, 0.0), (-m, 0.0), (0.0, m), (0.0, -m))):
            found = (dx, dy)
            break
    if found is None and bare is not None:
        logger.warning("auto-adjust: no offset within {:.2f} m solves with {:.0f}mm of slack "
                       "— taking the nearest bare pass", lim, m * 1000)
        found = bare
    if found is None:
        logger.warning("auto-adjust: pose ({:.3f},{:+.3f}) yaw {:+.1f}deg — NO chassis offset "
                       "within {:.2f} m makes its column reachable ({:.1f}s searched)",
                       px, py, float(np.rad2deg(pose[5])), lim, time.monotonic() - t0)
        return None
    dx, dy = found
    moved_dy = 0.0
    if abs(dx) >= cfg.CHASSIS_ADJUST_MIN_TRANSLATE_M:
        (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx))
    if abs(dy) >= cfg.CHASSIS_ADJUST_MIN_TRANSLATE_M:
        (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy))
        moved_dy = dy
    logger.info("auto-adjust: pose ({:.3f},{:+.3f}) -> turn {:+.1f}deg, dx {:+.3f} m, "
                "dy {:+.3f} m = nearest reachable spot ({:.1f}s searched)",
                px, py, turn, dx, dy, time.monotonic() - t0)
    return moved_dy


class ChassisNav:
    """--auto-move chassis LEG state, LEARNED within a run (in-memory; run()
    logs the final value for a manual config update). ONE instance per
    PHYSICAL leg being calibrated — the main source<->target leg, and
    (separately) the divert source<->divert-case leg — since they're
    different physical gaps with their own y-alignment references and must
    not teach each other.

    ONE shared distance per instance (`leg_dist_m`): the gap is a fixed
    physical distance, so every arrival residual — left or right — feeds the
    SAME number. (The main leg used to split this per-direction to calibrate
    out a direction-dependent travel gain, but that meant the very first
    residual ever measured — the operator's manual start park, no leg behind
    it — silently seeded the LEFT number before any left leg had run; a
    shared value plus the skip_next_learn guard below is the simpler fix.)
    Clamped to this instance's construction distance +/- CHASSIS_LEG_LEARN_CLAMP_M.

    `cur_ref` tracks the y THIS leg's chassis is currently aligned to — a
    case/battery grab-point centering ref OR a bin-center align target,
    whichever ran last for this leg. None until the first alignment (nothing
    legitimate to subtract yet, so that first ref_change is deliberate-free).
    Centering refs differ per ITEM/slot, so a re-alignment contains a
    DELIBERATE component (ref_change) that must not teach the leg — only the
    arrival residual does. Turns and manual (keyboard) corrections are not
    tracked.

    `skip_next_learn`: True by construction (`skip_first`) for the main leg
    — its first-ever centering measures the OPERATOR's manual start park, not
    any leg's arrival. Pass `skip_first=False` for a leg whose first use IS a
    real, just-executed strafe (e.g. the divert leg). Also set explicitly at
    excursion call sites whose arrival isn't measured against this leg."""

    def __init__(self, base_dist_m: "float | None" = None, skip_first: bool = True) -> None:
        self._base_dist_m = float(cfg.CHASSIS_AUTO_STRAFE_DIST_M if base_dist_m is None
                                  else base_dist_m)
        self.leg_dist_m = self._base_dist_m
        self.cur_ref: "float | None" = None
        self.skip_next_learn = skip_first

    def dist(self, direction: str) -> float:
        return self.leg_dist_m

    def ref_change(self, new_ref: float) -> float:
        """Deliberate strafe component of centering to `new_ref` (old - new,
        +left); updates the tracked ref. 0.0 when the ref is unchanged, or
        when this is the first-ever alignment (cur_ref is None — nothing
        legitimate to subtract)."""
        d = 0.0 if self.cur_ref is None else self.cur_ref - float(new_ref)
        self.cur_ref = float(new_ref)
        return d

    def learn(self, dy: float, at: str) -> None:
        """Feed one centering/adjust RESIDUAL (m, +left; deliberate ref moves
        already removed) at `at` ("source"/"divert" = after a LEFT-direction
        leg, "target" = after a RIGHT-direction leg) into this leg's distance."""
        if dy == 0.0:
            return
        lo = self._base_dist_m - cfg.CHASSIS_LEG_LEARN_CLAMP_M
        hi = self._base_dist_m + cfg.CHASSIS_LEG_LEARN_CLAMP_M
        # LEFT arrival landed long -> shorten; RIGHT arrival had to go further -> lengthen
        delta = dy if at in ("source", "divert") else -dy
        # DAMPED: at unity gain this tracks the last residual instead of
        # averaging them, which is why the divert leg (few samples, +-100mm
        # residuals) swung 246mm across one run — see CHASSIS_LEG_LEARN_GAIN
        new = float(np.clip(self.leg_dist_m + float(cfg.CHASSIS_LEG_LEARN_GAIN) * delta,
                            lo, hi))
        logger.info("leg distance: {:.3f} -> {:.3f} m ({} residual {:+.3f}, "
                    "gain {:.2f})", self.leg_dist_m, new, at, dy,
                    float(cfg.CHASSIS_LEG_LEARN_GAIN))
        self.leg_dist_m = new


def _center_case(bot, layers: int, label: str, station: str,
                 nav: "ChassisNav | None", y_ref: float,
                 plane_z: "float | None" = None,
                 tol_m: "float | None" = None,
                 min_turn_deg: "float | None" = None,
                 max_moves: "int | None" = None,
                 x_ref: "float | None" = None):
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
    ``x_ref``: also drive FORWARD/BACK until the case center sits at this x
    (post-turn frame, same clamps as the strafe). None = leave x alone, the old
    behaviour. Without it the alignment fixes y and yaw and leaves x wherever
    the operator or the last leg put it, so a station parked 20cm out in x
    passed centering and then failed the DESCENT pre-check, and _auto_adjust
    recovered it with one big lunge (0903 20:50: centering finished, case still
    at x=1.089 against a 0.87 ref, pre-check failed 20.6mm short, then a single
    Move FORWARD 0.219m). Aligning x here spreads that over the same
    re-detect/correct rounds as y and mostly stops the pre-check failing at all.

    Only the STRAFE teaches the legs: they are left/right distances, so an x
    move is not evidence about them.

    Strafes feed the learned per-direction leg distance MINUS the deliberate
    item-to-item ref change (nav.ref_change) so re-alignments don't teach the
    legs; turns are not tracked. Returns the LAST detection (a None /
    not-found detection returns immediately for the caller's normal handling;
    already-centered costs exactly one detect). `tol_m` overrides
    cfg.CHASSIS_CENTER_TOL_M for callers that want a tighter/looser deadband."""
    from .move_chassis import (move_backward, move_forward, strafe_left,
                               strafe_right, turn_ccw, turn_cw)
    skip_learn = False
    if nav is not None:
        skip_learn = nav.skip_next_learn
        nav.skip_next_learn = False
        if skip_learn:
            logger.info("[{}] arrival at {} unattributable (divert/blind place) — "
                        "correcting position without teaching the legs", label, station)
    tol = cfg.CHASSIS_CENTER_TOL_M if tol_m is None else tol_m
    min_turn = (cfg.CHASSIS_ADJUST_MIN_TURN_DEG if min_turn_deg is None
                else float(min_turn_deg))
    rounds = int(cfg.CHASSIS_CENTER_MAX_MOVES if max_moves is None else max_moves)
    if tol_m is not None or min_turn_deg is not None:
        logger.info("[{}] centering at {} with STRICT deadbands: {:.0f}mm / "
                    "{:.1f}deg, up to {} rounds", label, station, tol * 1000.0,
                    min_turn, rounds)
    deliberate = nav.ref_change(y_ref) if nav is not None else 0.0
    # ONE arrival teaches the leg ONCE. Rounds 2+ of this loop re-measure the
    # SAME arrival after a correction, so they carry no new evidence about the
    # leg distance — they are the correction strafe's own error, plus whatever
    # deliberate ref change round 1 could not finish (round 1 is capped by
    # CHASSIS_ADJUST_MAX_TRANSLATE_M, and `deliberate` is zeroed after it).
    # Feeding them all double-counts: 0903 19:37 the divert leg took a -140mm
    # residual and then a -67mm one from a single arrival whose real error was
    # -140mm, moving the distance -207mm. The DIVERT leg is where this bites
    # because its y_ref jumps 160mm between the two slots (BAT_SRC_2 -0.050 vs
    # BAT_SRC_1 +0.110), so its round 1 hits the translate clamp and a round 2
    # always follows — the main leg's per-item refs are close enough to finish
    # in one round, which is why only the divert leg drifted.
    learned_once = False
    det = detect(bot, layers, plane_z)
    for _ in range(rounds):
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
        if abs(yaw) >= min_turn:
            turn = float(np.clip(yaw, -cfg.CHASSIS_ADJUST_MAX_TURN_DEG,
                                 cfg.CHASSIS_ADJUST_MAX_TURN_DEG))
        # case center in the post-turn base frame (frame rotated CCW by `turn`)
        th = float(np.deg2rad(turn))
        x, y = det.base_xy
        dy = float(-np.sin(th) * x + np.cos(th) * y) - y_ref
        dx = (0.0 if x_ref is None
              else float(np.cos(th) * x + np.sin(th) * y) - float(x_ref))
        if turn == 0.0 and abs(dy) <= tol and abs(dx) <= tol:
            return det
        if turn != 0.0:
            logger.info("[{}] centering case at {}: turn {} {:.1f} deg (case yaw {:+.1f})",
                        label, station, "ccw" if turn > 0 else "cw", abs(turn), yaw)
            (turn_ccw if turn > 0 else turn_cw)(bot, angle_deg=abs(turn))
        if abs(dx) > tol:
            dx = float(np.clip(dx, -cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M,
                               cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M))
            # +dx: the case is too far AHEAD, so close on it. Never teaches the
            # legs (they are the left/right gap).
            logger.info("[{}] centering case at {}: move {} {:.3f} m (ref x {:.3f})",
                        label, station, "forward" if dx > 0 else "back", abs(dx),
                        float(x_ref))
            (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx))
        if abs(dy) > tol:
            dy = float(np.clip(dy, -cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M,
                               cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M))
            logger.info("[{}] centering case at {}: strafe {} {:.3f} m (ref y {:+.3f})",
                        label, station, "left" if dy > 0 else "right", abs(dy), y_ref)
            (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy))
            if nav is not None and not skip_learn and not learned_once:
                nav.learn(dy - deliberate, station)
                learned_once = True
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


def _max_reachable_z(mover: SuctionMover, x: float, y: float, rpy) -> "float | None":
    """Highest EE height, from the current one down to
    DESCENT_CHECK_BOTTOM_EE_Z, where (x, y, z, *rpy) solves clear of
    collision — each height checked INDEPENDENTLY (fresh IK seed per z, no
    warm-chaining), because a collision at the top of the column does not
    mean every height below it is blocked too: descent_reachable bails on
    the FIRST failure, which for a wrist rotated to its lid-aligned angle can
    be the arm's current (near-transport) height alone, leaving the actual
    pick depth below it untested. None if nothing in the column is clear."""
    z0 = float(mover.current_ee_pose()[0][2])
    bottom = float(cfg.DESCENT_CHECK_BOTTOM_EE_Z)
    zs = np.arange(max(z0, bottom), bottom - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M))
    if zs[-1] > bottom + 1e-9:
        zs = np.append(zs, bottom)
    for z in zs:
        sol = mover.solve_pose((x, y, float(z)), rpy, seed=None)
        if sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision:
            return float(z)
    return None


# The session's right-arm mover, set by _main once it has built the
# GripperMover (a GripperMover IS the right arm's ArmMover). The lid stance /
# restore used to build a fresh right ArmMover each time (~3 s of model build
# per call, twice per lid task); with the cache they reuse the session's one.
# Standalone tools (lid_probe, lid_place) never set it and build as before.
_RIGHT_ARM: list = []


def _right_arm(bot) -> ArmMover:
    """The session's right-arm mover if _main built one, else a fresh ArmMover."""
    if _RIGHT_ARM:
        return _RIGHT_ARM[0]
    return ArmMover(robot=bot, side="right", ee_frame=cfg.GRIPPER_EE_FRAME)


def _preload_detectors() -> None:
    """Load the YOLO detectors the tasks use (case BEV, bin BEV, box BEV) once,
    at session start, so a task's first detection does not pay the load. Each
    loader is a once-only singleton in its module. A missing weight file is
    logged, not fatal — that task's own detection raises the same error if it
    is ever run."""
    import detect_bin as dbn          # case_detection siblings (path set at module import)
    import detect_box_bev as dbx
    for name, load in (("case BEV", dcb.load_model), ("bin BEV", dbn.load_bev_model),
                       ("box BEV", dbx.load_model)):
        t0 = time.monotonic()
        try:
            load()
            logger.info("preload: {} detector loaded ({:.1f}s)", name, time.monotonic() - t0)
        except Exception as e:                       # noqa: BLE001 — report, keep going
            logger.warning("preload: {} detector NOT loaded: {}", name, e)


def _view_park(mover: SuctionMover, label: str) -> None:
    """Move the arm out of the head-camera view: Cartesian park (keeps the
    current EE orientation — safe with or without a held item, tunable via
    config), joint park fallback if unset/unreachable.

    A below-transport start (the lift_to_clear picks/places stop at the
    wall-clear height) first rises STRAIGHT to SAFE_TRANSPORT_Z — the joint
    move to the park otherwise splits the rise across the whole path and
    sweeps sideways while still low (observed: headed left at ~0.95).
    Already-at-transport starts skip it (no-op)."""
    park = cfg.ARM_VIEW_PARK_EE_POS
    pos, rpy = mover.current_ee_pose()
    if pos[2] < cfg.SAFE_TRANSPORT_Z - 0.01:
        mover.move_ee_vertical(cfg.SAFE_TRANSPORT_Z, rpy)
    parked = False
    if park is not None:
        _, rpy = mover.current_ee_pose()
        parked = mover.move_ee(tuple(park), tuple(rpy)) is not None
        if not parked:
            # move_ee's single min-motion solve from the live q strands on the
            # start column's elbow branch (observed 49-113mm short, varying
            # with the start pose, INCLUDING targets inside points that
            # solved) — retry from the home seed, which reaches the park's
            # own branch, before giving up to the full joint park
            sol = mover.solve_pose(tuple(park), tuple(rpy),
                                   seed=np.asarray(cfg.ARM_VIEW_PARK_JOINTS,
                                                   dtype=np.float64),
                                   min_motion=False)
            if (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits
                    and not sol.in_collision):
                logger.info("[{}] view park reached via the home-seed branch "
                            "(min-motion solve was short)", label)
                mover.move_joints(sol.q)
                parked = True
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


def _detect_bin_xy(bot, n: "int | None" = None) -> "tuple[float, float] | None":
    """Bin detection on the metric BEV canvas (detect_bin.find_bin_bev, OBB),
    warped at DIVERT_BIN_PLANE_Z_M — the plane of the face the labels trace, so
    the box center maps linearly to base xy with none of the raw-frame
    projection bias. CLASS FILTERED to cfg.BIN_CLS_ID: the set is 2-class, and
    an unfiltered highest-conf pick can hand back the lid. Up to `n`
    fresh-frame attempts (default cfg.SEED_BIN_DETECT_N), combined by
    per-axis MEDIAN over the successful ones — one missed/jittery frame can't
    push the caller to its fallback or drag the center (mirrors _refine_det
    for cases). Returns None only when EVERY frame misses."""
    import detect_bin as dbn  # case_detection sibling (path set at module import)
    n = int(cfg.SEED_BIN_DETECT_N if n is None else n)
    pts: list[tuple[float, float]] = []
    yaws: list[float] = []
    for _ in range(max(1, n)):
        rgb = _head_rgb(bot)
        det = None if rgb is None else dbn.find_bin_bev(
            rgb, *_joints(bot), plane_z=cfg.DIVERT_BIN_PLANE_Z_M,
            cls_id=cfg.BIN_CLS_ID)
        if det is not None:
            pts.append((float(det[0]), float(det[1])))
            yaws.append(float(det[2]))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if len(pts) > 1:
        logger.info("bin detected on {}/{} frames: xy=({:.3f},{:+.3f}) "
                    "yaw={:.1f}deg (spread x {:.0f} / y {:.0f} mm)", len(pts), n,
                    float(np.median(xs)), float(np.median(ys)),
                    float(np.median(yaws)),
                    (max(xs) - min(xs)) * 1000, (max(ys) - min(ys)) * 1000)
    return float(np.median(xs)), float(np.median(ys))


def _bin_lid_detector(rgb, q_torso, q_head, plane_z):
    """The bin lid (class cfg.LID_CLS_ID on the 2-class bin OBB set)."""
    import detect_bin as dbn          # case_detection sibling
    return dbn.find_bin_bev(rgb, q_torso, q_head, plane_z=plane_z,
                            cls_id=cfg.LID_CLS_ID)


def _box_lid_detector(rgb, q_torso, q_head, plane_z):
    """The PAPER box lid (class cfg.BOX_LID_CLS_ID on the 2-class box OBB set).
    Same (x, y, yaw_deg) shape as the bin one; the detection's own size lands in
    _LAST_LID_DIMS for the side grasp, which needs the lid's short side."""
    import detect_box_bev as dbx      # case_detection sibling
    det = dbx.detect_box_bev(rgb, q_torso, q_head, plane_z=plane_z,
                             cls_id=cfg.BOX_LID_CLS_ID)
    if not det.found:
        return None
    _LAST_LID_DIMS.clear()
    _LAST_LID_DIMS.update(long=det.dims_m[0], short=det.dims_m[1], conf=det.conf)
    return det.base_xy[0], det.base_xy[1], det.base_yaw_deg


_LAST_LID_DIMS: dict = {}


def _detect_lid_xy(bot, n: "int | None" = None, plane_z: "float | None" = None,
                   detector=None, offset=None):
    """Lid detection on the metric BEV canvas -> (x, y, yaw_deg) in base, or
    None. Same shape as _detect_bin_xy (median over n fresh frames) but CLASS
    FILTERED to cfg.LID_CLS_ID and it also returns the yaw, which the pick
    needs for the wrist and for rotating the grab offset.

    ``plane_z``: the plane to WARP at (default cfg.LID_PLANE_Z_M). With
    cfg.LID_DEPTH_REFINE the warp plane no longer has to be the lid's true
    height: the ZED depth measures that under the detected center, the center is
    moved onto it exactly (bev.reproject_plane), and the measured height comes
    back as the 4th return value for the descent to use as its expected
    contact. Without depth (no stream, too few valid pixels) the plane is taken
    on faith as before and the 4th value is None.

    Returns (x, y, yaw_deg, surface_z | None)."""
    n = int(cfg.SEED_BIN_DETECT_N if n is None else n)
    plane_z = float(cfg.LID_PLANE_Z_M if plane_z is None else plane_z)
    # WARP at the configured plane (the scale the detector was trained near),
    # but SAMPLE depth from the last measured height when there is one — see
    # _LID_PLANE_SEEN. Two different jobs, two different numbers.
    guess = float(_LID_PLANE_SEEN.get(round(plane_z, 4), plane_z))
    pts: list[tuple[float, float, float]] = []
    find = _bin_lid_detector if detector is None else detector
    for _ in range(max(1, n)):
        rgb = _head_rgb(bot)
        det = None if rgb is None else find(rgb, *_joints(bot), plane_z)
        if det is not None:
            # keep the LAST frame that produced a detection, with the joints and
            # plane it was warped at, so save_lid_frame can write out the image
            # the accepted result actually came from instead of a fresh grab
            _LAST_LID_FRAME.clear()
            _LAST_LID_FRAME.update(rgb=rgb, joints=_joints(bot), plane_z=plane_z)
        if det is not None:
            pts.append((float(det[0]), float(det[1]), float(det[2])))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    raw_yaw = _median_yaw([p[2] for p in pts])
    yaws = cfg.canonical_lid_yaw(raw_yaw)   # one branch for every consumer
    x, y = float(np.median(xs)), float(np.median(ys))
    logger.info("lid detected on {}/{} frames (warped at z={:.3f}): xy=({:.3f},"
                "{:+.3f}) yaw={:.1f}deg{} (spread x {:.0f} / y {:.0f} mm)",
                len(pts), n, plane_z, x, y, yaws,
                "" if abs(yaws - raw_yaw) < 1e-6 else
                " (OBB said {:+.1f}, folded to the near-0 branch)".format(raw_yaw),
                (max(xs) - min(xs)) * 1000, (max(ys) - min(ys)) * 1000)
    if not cfg.LID_DEPTH_REFINE:
        return x, y, yaws, None
    if abs(guess - plane_z) > 1e-6:
        logger.info("depth sampling seeded at z={:.4f} (last measurement) rather "
                    "than the {:.3f} warp plane", guess, plane_z)
    return _refine_lid_with_depth(bot, x, y, yaws, plane_z, guess, offset)


def _refine_lid_with_depth(bot, x: float, y: float, yaws: float, plane_z: float,
                           guess: "float | None" = None, offset=None):
    """Move a lid detection onto the height the ZED depth measures under it.

    The warp plane and the surface height stopped being the same number (see
    cfg.LID_DEPTH_REFINE): this measures the surface, reprojects the center onto
    it, and hands the height back for the descent. Falls back to the unrefined
    detection — never raises — because a missing depth frame must not stop a
    run; the log says which happened."""
    try:
        import bev
        import depth_plane as dp

        f = dict(_LAST_LID_FRAME)
        rgb, joints = f.get("rgb"), f.get("joints")
        depth = bot.sensors.head_camera.get_depth()
        if rgb is None or depth is None:
            logger.warning("no depth frame — using the detection AS WARPED at "
                           "z={:.3f} (surface height unverified)", plane_z)
            return x, y, yaws, None
        g = float(plane_z if guess is None else guess)
        z_face, n_px, spread = dp.plane_from_depth(depth, rgb.shape, joints[0],
                                                   joints[1], (x, y), g)
        if z_face is None:
            logger.warning("depth window had only {} valid pixels — using the "
                           "detection AS WARPED at z={:.3f}", n_px, plane_z)
            return x, y, yaws, None
        C = bev.camera_centre(joints[0], joints[1])
        xy = bev.reproject_plane((x, y), plane_z, z_face, C)
        # The CONTACT height is measured under the CUP, not under the lid's
        # centre: the cup lands LID_GRAB_OFFSET_M away (rotated into the lid's
        # frame), and on a lid with ~20mm of embossing that offset can be the
        # difference between a ridge and a groove. The centre is still what the
        # reprojection uses — that is where the OBB centre lies.
        # the offset the CALLER grabs at — the bin lid's by default, (0,0) for
        # the paper lid, which is picked at its centre
        ox, oy = cfg.LID_GRAB_OFFSET_M if offset is None else offset
        c_, s_ = float(np.cos(np.deg2rad(yaws))), float(np.sin(np.deg2rad(yaws)))
        cup = (xy[0] + c_ * ox - s_ * oy, xy[1] + s_ * ox + c_ * oy)
        z_top, _, _ = dp.plane_from_depth(depth, rgb.shape, joints[0], joints[1],
                                          cup, z_face,
                                          pct=float(cfg.LID_DEPTH_CONTACT_PCT))
        z_top = z_face if z_top is None else z_top
        logger.info("DEPTH: surface under the lid is at z={:.4f} (warped at "
                    "{:.3f}, off by {:+.0f}mm; {} px, relief {:.0f}mm) -> center "
                    "({:.3f},{:+.3f}) -> ({:.3f},{:+.3f}), i.e. ({:+.0f},{:+.0f})mm",
                    z_face, plane_z, (z_face - plane_z) * 1000, n_px, spread * 1000,
                    x, y, xy[0], xy[1], (xy[0] - x) * 1000, (xy[1] - y) * 1000)
        logger.info("DEPTH: contact expected at ee_z {:.4f} (tallest points under "
                    "the CUP ({:.3f},{:+.3f}): z={:.4f}, + cup {:.3f})",
                    z_top + cfg.SUCTION_LENGTH_M, cup[0], cup[1], z_top,
                    cfg.SUCTION_LENGTH_M)
        _LID_PLANE_SEEN[round(float(plane_z), 4)] = float(z_face)
        return float(xy[0]), float(xy[1]), yaws, float(z_top)
    except Exception as e:  # noqa: BLE001 — a refinement must not stop a run
        logger.warning("depth refinement failed ({}) — using the detection as "
                       "warped at z={:.3f}", e, plane_z)
        return x, y, yaws, None


def _center_lid(bot, plane_z: "float | None" = None,
                x_ref: "float | None" = None, y_ref: "float | None" = None,
                x_min: "float | None" = None, detector=None, offset=None):
    """Chassis alignment to a lid: turn it square and drive until the CUP point
    (its center + the rotated grab offset) sits at the reference xy. Returns the
    last detection (x, y, yaw_deg), or None.

    Defaults are the PICK side (cfg.LID_PLANE_Z_M, cfg.LID_CENTER_XY_M); pass
    ``plane_z`` + refs to align to the lid on the FLOOR at the unload station
    instead (cfg.LID_FLOOR_PLANE_Z_M) — same arithmetic, different plane.

    ``x_min``: a hard floor on the cup point's x — no forward move may end
    closer than this, whatever the reference says. Redundant while
    x_ref >= x_min, which is the point: closing on the drop-off is the one
    correction with something in front of the robot, so an open-loop leg that
    overshoots, or a reference someone lowers later, must not walk into it.

    _center_case's job, but standalone: that one is built around the case
    detector's det object and the ChassisNav leg learning, and lid mode has
    neither (the operator hand-drives both legs, so there is no repeated leg to
    teach). Same arithmetic though — turn first, then compute the drive in the
    POST-TURN frame, clamp per move, re-detect, up to N rounds. Deadbands come
    from the strict start-alignment constants, since this IS lid mode's start
    alignment. It aligns the cup point rather than the lid center because that
    is what has to be reachable; with LID_GRAB_OFFSET_M small the two nearly
    coincide anyway."""
    from .move_chassis import (move_backward, move_forward, strafe_left,
                               strafe_right, turn_ccw, turn_cw)
    if x_ref is None or y_ref is None:
        x_ref, y_ref = (float(v) for v in cfg.LID_CENTER_XY_M)
    x_ref, y_ref = float(x_ref), float(y_ref)
    plane_z = float(cfg.LID_PLANE_Z_M if plane_z is None else plane_z)
    tol = float(cfg.CHASSIS_START_CENTER_TOL_M)
    min_turn = float(cfg.CHASSIS_START_MIN_TURN_DEG)
    rounds = int(cfg.CHASSIS_START_CENTER_MAX_MOVES)
    ox, oy = cfg.LID_GRAB_OFFSET_M if offset is None else offset
    logger.info("aligning the chassis to the lid: cup point -> ({:.3f},{:+.3f}), "
                "deadband {:.0f}mm / {:.1f}deg, up to {} rounds{}",
                x_ref, y_ref, tol * 1000.0, min_turn, rounds,
                "" if x_min is None else ", x floor {:.3f}".format(float(x_min)))
    det = _detect_lid_xy(bot, plane_z=plane_z, detector=detector, offset=offset)
    for _ in range(rounds):
        if det is None:
            return None
        lx, ly, lyaw_deg = det[0], det[1], det[2]
        c, sn = float(np.cos(np.deg2rad(lyaw_deg))), float(np.sin(np.deg2rad(lyaw_deg)))
        gx, gy = lx + c * ox - sn * oy, ly + sn * ox + c * oy
        if abs(gy - y_ref) > cfg.CHASSIS_DETECT_Y_GATE_M:
            # class 1 is "a lid" — the one on the FLOOR at the unload spot can
            # be in frame too, and centering on that drives the robot away.
            logger.warning("lid detection rejected: cup y {:+.3f} vs expected {:+.3f} "
                           "(> {:.2f} m gate — the unload-side lid?)", gy, y_ref,
                           cfg.CHASSIS_DETECT_Y_GATE_M)
            return None
        yaw = lyaw_deg - 180.0 if lyaw_deg >= 90.0 else lyaw_deg   # long-axis wrap
        turn = (float(np.clip(yaw, -cfg.CHASSIS_ADJUST_MAX_TURN_DEG,
                              cfg.CHASSIS_ADJUST_MAX_TURN_DEG))
                if abs(yaw) >= min_turn else 0.0)
        th = float(np.deg2rad(turn))
        dy = float(-np.sin(th) * gx + np.cos(th) * gy) - y_ref
        dx = float(np.cos(th) * gx + np.sin(th) * gy) - x_ref
        if turn == 0.0 and abs(dy) <= tol and abs(dx) <= tol:
            logger.info("lid aligned: cup ({:.3f},{:+.3f}) yaw {:+.1f}deg", gx, gy, yaw)
            return det
        if turn != 0.0:
            logger.info("lid align: turn {} {:.1f}deg (lid yaw {:+.1f})",
                        "ccw" if turn > 0 else "cw", abs(turn), yaw)
            (turn_ccw if turn > 0 else turn_cw)(bot, angle_deg=abs(turn))
        if abs(dx) > tol:
            dx = float(np.clip(dx, -cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M,
                               cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M))
            if x_min is not None and dx > 0.0:
                # +dx closes the gap, so the cup would end at (its x - dx)
                room = float(np.cos(th) * gx + np.sin(th) * gy) - float(x_min)
                if dx > room:
                    logger.warning("lid align: forward move {:.3f} m clamped to "
                                   "{:.3f} m — cup x floor {:.3f}", dx,
                                   max(room, 0.0), float(x_min))
                    dx = max(room, 0.0)
            if abs(dx) > tol:
                logger.info("lid align: move {} {:.3f} m (ref x {:.3f})",
                            "forward" if dx > 0 else "back", abs(dx), x_ref)
                (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx))
        if abs(dy) > tol:
            dy = float(np.clip(dy, -cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M,
                               cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M))
            logger.info("lid align: strafe {} {:.3f} m (ref y {:+.3f})",
                        "left" if dy > 0 else "right", abs(dy), y_ref)
            (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy))
        det = _detect_lid_xy(bot, plane_z=plane_z, detector=detector,
                             offset=offset)
    logger.warning("lid alignment used all {} rounds — proceeding with the last "
                   "detection", rounds)
    return det


_LAST_LID_FRAME: dict = {}
# Last depth-measured surface height, keyed by the CONFIG plane it was found
# from. The warp plane is only a guess, and a wrong guess walks the first depth
# sample pixel across the surface by |P - camera_xy| * dz / (z - camera_z):
# 0904's 200mm plane error moved it 176mm, against a lid half-width of ~175mm —
# it landed on the lid by luck. Seeding the next detection with what was
# measured last time drives that walk to ~0, so the FIRST sample is already on
# the surface. Keyed per plane so the pick (box lid) and the place (floor lid)
# do not seed each other.
_LID_PLANE_SEEN: dict = {}


def save_lid_frame(bot, tag: str, lid_xy=None, cup_xy=None, yaw_deg=None) -> None:
    """Write the lid detection's own frame next to the run log, plus its BEV
    warp with the geometry the place is about to use drawn on it.

    Called once the place is COMMITTED — the pose resolved and the column
    checked — so the run log's numbers have the picture that produced them
    beside them, and a place that lands wrong can be diagnosed after the fact
    instead of re-staged. Goes to cfg.LID_IMAGE_DIR under the run stamp. Uses
    the frame the accepted detection came from (_LAST_LID_FRAME); falls back to
    a fresh grab if that is empty. Never raises: a failed screenshot must not
    stop a place."""
    if cfg.LID_IMAGE_DIR is None:
        return
    try:
        import cv2
        import bev
        import config as dcfg          # case_detection's own config (BEV canvas)

        f = dict(_LAST_LID_FRAME)
        rgb = f.get("rgb")
        joints, plane_z = f.get("joints"), f.get("plane_z")
        if rgb is None:
            rgb, joints, plane_z = _head_rgb(bot), _joints(bot), cfg.LID_PLANE_Z_M
            logger.warning("no stored detection frame — saving a fresh grab")
        if rgb is None:
            logger.warning("no head frame to save")
            return
        Path(cfg.LID_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        base = Path(cfg.LID_IMAGE_DIR) / f"run_{run_stamp()}_{tag}"
        cv2.imwrite(str(base) + ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        canvas = cv2.cvtColor(bev.build_mapper(*joints, float(plane_z)).warp(rgb),
                              cv2.COLOR_RGB2BGR)
        x0, y0 = dcfg.BEV_X_RANGE[0], dcfg.BEV_Y_RANGE[0]
        s = float(dcfg.BEV_PX_PER_M)

        def px(x, y):                  # base xy -> BEV pixel (inverse of bev_px_to_base)
            return int(round((float(x) - x0) * s)), int(round((float(y) - y0) * s))

        if lid_xy is not None:
            cv2.circle(canvas, px(*lid_xy), 8, (0, 255, 255), 2)
            if yaw_deg is not None:
                t = float(np.deg2rad(yaw_deg))
                a = px(lid_xy[0] + 0.15 * np.cos(t), lid_xy[1] + 0.15 * np.sin(t))
                b = px(lid_xy[0] - 0.15 * np.cos(t), lid_xy[1] - 0.15 * np.sin(t))
                cv2.line(canvas, a, b, (0, 255, 255), 2)
        if cup_xy is not None:
            u, v = px(*cup_xy)
            cv2.drawMarker(canvas, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.imwrite(str(base) + "_bev.png", canvas)
        logger.info("saved {}/{}.png and _bev.png (cross = the cup target, "
                    "circle + line = the detected lid)", cfg.LID_IMAGE_DIR,
                    base.name)
    except Exception as e:  # noqa: BLE001 — diagnostics must never kill a place
        logger.warning("could not save the lid frame ({}): {}", tag, e)


def _median_yaw(yaws: list) -> float:
    """Median yaw with the same 180-deg unwrap _refine_det uses: the OBB long
    axis is symmetric under a flip, so raw samples can straddle the branch and
    a plain median would land between them."""
    out = [float(yaws[0])]
    for y in yaws[1:]:
        y = float(y)
        while y - out[0] > 90.0:
            y -= 180.0
        while y - out[0] < -90.0:
            y += 180.0
        out.append(y)
    return float(np.median(out))


def _lid_pick_aim(det, offset=None, plane=None):
    """(cup_x, cup_y, expected_ee_z) for a lid detected on the box. The grab
    offset is in the LID's own frame, exactly as resolve_poses rotates
    CASE_GRAB_OFFSET by the detected case yaw.

    The contact height is det[3] — the surface _refine_lid_with_depth MEASURED
    under the cup — falling back to the LID_PLANE_Z_M guess only when there was
    no depth to measure. This is the SINGLE source of the pick's ez: it used to
    live here as the guess and be corrected inside _pick_yaw_search, so run_lid,
    which calls this function again for its own gx/gy, silently handed the
    UNCORRECTED z to pick(). 0905 15:30 is what that costs — depth measured the
    lid 93mm below the guessed plane, and the reprojection that same correction
    drives pushed the cup 85mm further out in x. run_lid took the far xy with
    the high z (creep line 1.0950 instead of 1.0096), i.e. the one combination
    that extends the arm most, and 2.7mm into the creep the IK went singular
    (joint step 2.2x the per-tick cap for a 0.18mm Cartesian step, held 101
    ticks) — the pick died as 'unreachable' before it ever touched the lid.
    """
    lx, ly, lyaw_deg = det[0], det[1], det[2]
    yaw = float(np.deg2rad(lyaw_deg))
    # offset / plane default to the BIN lid's; --box-lid passes the paper lid's,
    # which is grabbed at its CENTRE and sits at its own height
    dx, dy = cfg.LID_GRAB_OFFSET_M if offset is None else offset
    c, sn = float(np.cos(yaw)), float(np.sin(yaw))
    z_face = (float(det[3]) if len(det) > 3 and det[3] is not None
              else float(cfg.LID_PLANE_Z_M if plane is None else plane))
    return (lx + c * dx - sn * dy, ly + sn * dx + c * dy,
            z_face + float(cfg.SUCTION_LENGTH_M))


def _nearest_pick_spot(mover, det, margin: float = 0.0):
    """The cup xy CLOSEST to where the lid is that the PICK column solves at
    (the one lid-aligned wrist yaw _pick_yaw_search flies), or None — the
    pick-side twin of _nearest_place_spot, so run_lid's chassis correction is
    the smallest move that fixes reach instead of a drive to LID_CENTER_XY_M
    (which dragged both axes to a fixed point). Same column as
    _pick_yaw_search's descent_reachable: from the arm's current height down
    to DESCENT_CHECK_BOTTOM_EE_Z. ``margin``: the spot must also solve with
    the cup that far off in +-y and the column that much longer at both ends
    (_column_ok_quiet), so the open-loop move does not land on the edge of
    reach. Never closer than LID_CENTER_MIN_X_M."""
    gx, gy, _ez = _lid_pick_aim(det)
    w = float((np.deg2rad(det[2]) + cfg.GRASP_YAW + np.pi) % (2.0 * np.pi) - np.pi)
    hi = float(mover.current_ee_pose()[0][2])
    pz = float(cfg.DESCENT_CHECK_BOTTOM_EE_Z)
    x_min = float(cfg.LID_CENTER_MIN_X_M)
    step = float(cfg.CHASSIS_ADJUST_STEP_M)
    n = int(round(float(cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M) / step))
    cands = sorted(((i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
                    if abs(i) + abs(j) <= n),
                   key=lambda o: (abs(o[0]) + abs(o[1]), abs(o[0]), abs(o[1])))
    for ox, oy in cands:
        x, y = gx + ox, gy + oy
        if x < x_min - 1e-9:
            continue                      # never aim closer than the safety floor
        if not _column_ok_quiet(mover, x, y, w, hi, pz, margin):
            continue
        if margin > 0.0 and not all(
                _column_ok_quiet(mover, x, y + hy, w, hi, pz, margin)
                for hy in (margin, -margin)):
            continue
        logger.info("nearest solvable PICK cup xy is ({:.3f},{:+.3f}) — {:.0f}mm "
                    "forward/back and {:.0f}mm sideways from where the lid is "
                    "({:.3f},{:+.3f}), margin {:.0f}mm", x, y, ox * 1000, oy * 1000,
                    gx, gy, margin * 1000)
        return x, y
    return None


def _pick_yaw_search(mover, det, offset=None, plane=None, drop_to_reachable=False):
    """(pose, rpy, yaw_delta) for the lid pick, or (None, None, 0.0) if no wrist
    yaw solves the column where the lid currently is.

    ``drop_to_reachable`` (box-lid only, default off): if the column does not
    solve from wherever the arm currently is, descent_reachable's failure can
    be entirely about the CURRENT height, not the pick depth — it bails on
    the first z it checks, which is the current one. Scan down for the
    highest height this same (xy, wrist) pose is collision-free at
    (_max_reachable_z) and move there before giving up, so a column that was
    never actually tested below the arm's resting height gets its chance.
    UNVERIFIED on the robot.

    The wrist takes the LID's OWN yaw and nothing else: the canonical
    L + GRASP_YAW, or its 180 flip, which the lid is symmetric under and so
    grabs identically. Only those two.

    It used to walk outward from there — +-10, 20, 30, 45, 60, 75, 90 deg and
    each one's 180 partner, 30 candidates — on the reasoning that wrist yaw is
    FREE at a suction pick, the cup being round and the grab POINT coming from
    the lid's frame. That is true of the PICK and false of the run: the delta is
    carried to the place, which re-adds it, so a pick rescued at +45 asks the
    place for a wrist of L_floor + GRASP_YAW + 45. The place gets two branches
    (wyaw, wyaw + 180) inside the LEANED stance, whose reachable wrist band is
    narrow — see the note there about it excluding yaw near 0. So the wide
    search bought picks that could not be placed, converting a clean "align the
    chassis and pick straight" into a lid stuck on the cup at the unload spot.

    Cost of dropping it: a column that solves at neither branch now fails the
    search, which sends run_lid to the chassis alignment (its intended remedy)
    and, if that does not help either, stops before moving the arm. That is the
    better failure — it happens with an EMPTY cup.

    With the search this narrow the returned delta is always 0 or 180, and the
    place tries both branches anyway, so it no longer carries information. It is
    still returned correctly (it does describe the held lid's relation to the
    wrist) rather than zeroed, in case the place's branch pair ever narrows."""
    gx, gy, ez = _lid_pick_aim(det, offset, plane)  # ez is the measured surface
    yaw = float(np.deg2rad(det[2]))
    w = float((yaw + cfg.GRASP_YAW + np.pi) % (2.0 * np.pi) - np.pi)
    r = (float(cfg.GRASP_ORIENTATION_RPY[0]),
         float(cfg.GRASP_ORIENTATION_RPY[1]), w)
    if descent_reachable(mover, (gx, gy, ez, *r)):
        logger.info("lid pick wrist yaw {:+.1f}deg (lid yaw {:+.1f} + GRASP_YAW "
                    "{:+.1f})", float(np.rad2deg(w)), float(np.rad2deg(yaw)),
                    float(np.rad2deg(cfg.GRASP_YAW)))
        return (gx, gy, ez, *r), r, 0.0
    logger.warning("the pick column at ({:.3f},{:+.3f}) {:.3f}->{:.3f} does not "
                   "solve at the lid-aligned wrist {:+.1f}deg", gx, gy, ez,
                   float(cfg.DESCENT_CHECK_BOTTOM_EE_Z), float(np.rad2deg(w)))
    if drop_to_reachable:
        z_cur = float(mover.current_ee_pose()[0][2])
        z_new = _max_reachable_z(mover, gx, gy, r)
        if z_new is not None and z_new < z_cur - 1e-3:
            logger.warning("dropping to the highest reachable height {:.3f} "
                           "(from {:.3f}) before giving up on this column",
                           z_new, z_cur)
            if mover.move_ee((gx, gy, z_new), r, quiet=False) is not None:
                if descent_reachable(mover, (gx, gy, ez, *r)):
                    logger.info("lid pick wrist yaw {:+.1f}deg (lid yaw {:+.1f} + "
                                "GRASP_YAW {:+.1f}) — solved after the height drop",
                                float(np.rad2deg(w)), float(np.rad2deg(yaw)),
                                float(np.rad2deg(cfg.GRASP_YAW)))
                    return (gx, gy, ez, *r), r, 0.0
                logger.warning("still does not solve after dropping to {:.3f}", z_new)
            else:
                logger.warning("could not even reach the height-drop pose at {:.3f}",
                               z_new)
        else:
            logger.warning("no lower height on this column is collision-free either")
    return None, None, 0.0


def _lid_place_aim(tgt, yaw_delta: float):
    """(cup_x, cup_y, wrist_yaw) the place would fly for a floor-lid detection.
    The cup goes over the same point of the lid it grabbed, so the grab offset
    is rotated by the TARGET lid's yaw; the wrist re-adds the pick's delta so
    the lid lands aligned."""
    tx, ty, tyaw_deg = tgt[0], tgt[1], tgt[2]
    tyaw = float(np.deg2rad(tyaw_deg))
    dx, dy = cfg.LID_GRAB_OFFSET_M
    c, sn = float(np.cos(tyaw)), float(np.sin(tyaw))
    return (tx + c * dx - sn * dy, ty + sn * dx + c * dy,
            tyaw + float(cfg.GRASP_YAW) + float(yaw_delta))


def _column_ok_quiet(mover, x: float, y: float, wyaw: float,
                     hi: float, pz: float, margin: float = 0.0) -> bool:
    """Column hi -> pz at (x, y) at the ONE lid-aligned wrist yaw, warm-chained,
    silent. column_reachable logs a warning per failure, which is right for a
    pre-flight and wrong for a search over dozens of candidate spots.

    One branch, not two: the place flies ``wyaw`` and nothing else, so testing
    wyaw + 180 here would pass spots the place then refuses.

    ``margin`` extends the column at BOTH ends, so the answer holds even if the
    surface height re-measures that much higher or lower after the chassis
    moves (see cfg.LID_PLACE_SPOT_MARGIN_M)."""
    hi, pz = float(hi) + float(margin), float(pz) - float(margin)
    cand = float((wyaw + np.pi) % (2.0 * np.pi) - np.pi)
    r = (float(cfg.GRASP_ORIENTATION_RPY[0]),
         float(cfg.GRASP_ORIENTATION_RPY[1]), cand)
    seed = None
    for z in np.arange(hi, pz - 1e-9, -float(cfg.DESCENT_CHECK_STEP_M)):
        sol = mover.solve_pose((float(x), float(y), float(z)), r,
                               seed=seed, min_motion=seed is not None)
        if not (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits
                and not sol.in_collision):
            return False
        seed = sol.q
    return True


def _place_column_ok(mover, tgt, yaw_delta: float, sz: float, pz: float) -> bool:
    """Does the place column solve where the lid ALREADY is, on either wrist
    branch? The question the chassis correction should be asked, instead of
    'are we on the taught reference'."""
    px, py, wyaw = _lid_place_aim(tgt, yaw_delta)
    if _column_ok_quiet(mover, px, py, wyaw, sz, pz):
        logger.info("lid is at cup ({:.3f},{:+.3f}); the column {:.3f}->{:.3f} "
                    "already solves — no chassis move", px, py, sz, pz)
        return True
    return False


def _nearest_place_spot(mover, tgt, yaw_delta: float, sz: float, pz: float,
                        x_min: float, margin: float = 0.0):
    """The cup xy CLOSEST to where the lid is that the place column solves at,
    or None. Searched nearest-first over a grid of offsets.

    ``margin`` demands slack rather than a bare pass: the column must also
    solve with the surface that much higher and lower, and with the cup that
    far off in +/-Y. Without it the search returns a spot on the EDGE of the
    reachable region, which is what broke 0906 10:25. Not applied in x — that
    band is only ~40mm wide here and cannot pay for it (see
    cfg.LID_PLACE_SPOT_MARGIN_M); x drift is the retry rounds' job. The two
    halo columns are only checked once the candidate itself passes, so the
    common case (a candidate that fails outright) costs nothing extra.

    The alignment reference used to be a single taught point, so any correction
    dragged BOTH axes to it: 0904 a lid at x 0.991 (48mm short at the start
    height, so a correction was genuinely needed) also got strafed 300mm
    sideways because its y was -0.04 against a +0.30 reference, when -0.04 was
    perfectly reachable. Aiming at the nearest solvable spot instead makes the
    move the smallest one that fixes the actual problem."""
    px, py, wyaw = _lid_place_aim(tgt, yaw_delta)
    step, reach = 0.03, 0.30
    n = int(reach / step)
    cands = sorted(((dx * step, dy * step) for dx in range(-n, n + 1)
                    for dy in range(-n, n + 1)),
                   key=lambda o: (abs(o[0]) + abs(o[1]), abs(o[0]), abs(o[1])))
    for ox, oy in cands:
        x, y = px + ox, py + oy
        if x < x_min - 1e-9:
            continue                      # never aim closer than the safety floor
        if not _column_ok_quiet(mover, x, y, wyaw, sz, pz, margin):
            continue
        if margin > 0.0 and not all(
                _column_ok_quiet(mover, x, y + hy, wyaw, sz, pz, margin)
                for hy in (margin, -margin)):
            continue
        logger.info("nearest solvable cup xy is ({:.3f},{:+.3f}) — {:.0f}mm "
                    "forward/back and {:.0f}mm sideways from where the lid is "
                    "({:.3f},{:+.3f}), margin {:.0f}mm", x, y, ox * 1000,
                    oy * 1000, px, py, margin * 1000)
        return x, y
    return None


def lid_place_stance(bot, mover) -> None:
    """Put the robot in the lid PLACE stance: torso to cfg.LID_PLACE_TORSO_DEG
    and the arm to cfg.LID_UNLOAD_STOW_JOINTS, moving BOTH AT ONCE.

    Its own function so ik_demo.lid_probe can reproduce exactly the state the
    place happens in — a probe measured in a different stance measures nothing.

    Together, because the torso carries the arm base: doing them in sequence
    would drag a held lid through the pose a lean-with-a-frozen-arm produces.
    The arm goes by JOINT command (during the lean there is no fixed base to aim
    a Cartesian target at, and the taught config's elbow branch is the checked
    one), clipped into the IK band first.

    _park_during_legs runs the arm move in a background thread while the main
    thread drives the torso. The torso goes via move_torso — motion only — and
    NOT pin_torso, because pin_torso rebuilds the pinocchio model at the end and
    doing that under a running arm stream would swap the model out from under
    it. pin_torso is called once afterwards: by then the move is a no-op and all
    it does is re-model at the stance reached."""
    torso_target = np.deg2rad(np.asarray(cfg.LID_PLACE_TORSO_DEG, dtype=float))
    q_stow = mover.clip_to_band(cfg.LID_UNLOAD_STOW_JOINTS, "LID_UNLOAD_STOW_JOINTS")
    moves = [lambda: mover.move_joints(q_stow)]
    if cfg.LID_UNLOAD_RIGHT_JOINTS is not None:
        # The RIGHT arm goes to its own taught park in the same breath: the torso
        # carries both arm bases, so a right arm left wherever the last run put
        # it swings with the lean too. The session's right mover when _main
        # built one (_right_arm), else a fresh ArmMover (~3 s model build).
        right = _right_arm(bot)
        q_right = right.clip_to_band(cfg.LID_UNLOAD_RIGHT_JOINTS,
                                     "LID_UNLOAD_RIGHT_JOINTS")
        moves.append(lambda: right.move_joints(q_right))
    logger.info("=== torso -> {} deg AND both arms -> their stow joints, "
                "together ===", cfg.LID_PLACE_TORSO_DEG)
    _park_during_legs("lid", moves,
                      lambda: move_torso(bot.torso, torso_target,
                                         float(cfg.LID_TORSO_VEL_SCALE), 60.0))
    mover.pin_torso(torso_target)          # arrives already there: re-models only


def _lid_frame(cup_pos, lid_yaw_deg: float, short: float):
    """The lid as the GRIPPER sees it, from where the cup ended up.

    The cup took the lid at its CENTRE, so the centre sits right under the cup
    one suction length down. Returns (edge, approach, long_axis):

      * approach  = the lid's own +y, the direction the gripper travels in
      * edge      = the midpoint of the lid's -y side, where the fingers go
      * long_axis = the lid's own +x, for saying how far off centre a grasp is

    Shared with box_lid_jog, which is where these numbers were flown by hand."""
    lam = float(np.deg2rad(lid_yaw_deg))
    approach = np.array([-np.sin(lam), np.cos(lam), 0.0])
    long_axis = np.array([np.cos(lam), np.sin(lam), 0.0])
    centre = np.array([float(cup_pos[0]), float(cup_pos[1]),
                       float(cup_pos[2]) - float(cfg.SUCTION_LENGTH_M)])
    return centre - approach * (short / 2.0), approach, long_axis


def _side_rpy(lam_deg: float):
    """cfg.BOX_LID_SIDE_RPY spun about base z by ``lam_deg``, so the fingers stay
    parallel to the edge they close on.

    That rpy has pitch -90 deg, which is exactly gimbal lock for xyz euler — but
    its yaw stays pinned at 0 and the ROLL absorbs the spin (roll = -90 + spin),
    so the round trip through as_euler is exact (checked to 1e-4 deg over
    +-30 deg) and every rpy tuple downstream stays a plain xyz euler."""
    R = (Rotation.from_euler("z", float(np.deg2rad(lam_deg)))
         * Rotation.from_euler("xyz", np.asarray(cfg.BOX_LID_SIDE_RPY, dtype=float)))
    return tuple(float(v) for v in R.as_euler("xyz"))


def _box_lid_pick(bot, mover: SuctionMover) -> "tuple | None":
    """Detect the paper lid where the robot stands, pick it at its CENTRE with
    the gentle cardboard force pair, and lift to transport height.

    Returns (detection, short_side_m) or None. Split out of run_box_lid so
    box_lid_jog can rehearse the handoff after EXACTLY the pick the sequence
    flies — a second copy of it there would drift."""
    plane = float(cfg.BOX_LID_PLANE_Z_M)
    offset = tuple(float(v) for v in cfg.BOX_LID_GRAB_OFFSET_M)
    # Loop so an operator chassis nudge (unreachable pick column, no detection,
    # or an unreachable pick pose) re-detects and retries instead of giving up
    # after one shot — same "adjust the chassis, `d` to retry, `q` to give up"
    # gate the case/battery flow uses for the same failure shape.
    while True:
        det = _detect_lid_xy(bot, plane_z=plane, detector=_box_lid_detector,
                             offset=offset)
        pose, rpy, _ = ((None, None, 0.0) if det is None
                        else _pick_yaw_search(mover, det, offset, plane,
                                              drop_to_reachable=True))
        if det is not None and pose is None:
            logger.info("no wrist yaw solves the pick column where the lid is — "
                        "aligning the chassis to ({:.3f},{:+.3f})", *cfg.LID_CENTER_XY_M)
            det = _center_lid(bot, plane_z=plane, x_min=float(cfg.LID_CENTER_MIN_X_M),
                              detector=_box_lid_detector, offset=offset)
            if det is not None:
                pose, rpy, _ = _pick_yaw_search(mover, det, offset, plane,
                                                drop_to_reachable=True)
        if det is None:
            logger.warning("paper box lid NOT detected (box OBB class {}, warp plane "
                           "z={:.3f}) — adjust the chassis (f/b/l/r), `d` to re-detect "
                           "+ retry, `q` to give up", cfg.BOX_LID_CLS_ID, plane)
            if not _manual_strafe(bot, "adjust"):
                logger.error("paper box lid NOT detected — giving up, nothing picked")
                return None
            continue
        if pose is None:
            logger.warning("no wrist yaw solves the paper lid's pick column — adjust "
                           "the chassis (f/b/l/r), `d` to re-detect + retry, `q` to "
                           "give up")
            if not _manual_strafe(bot, "adjust"):
                logger.error("no wrist yaw solves the pick column — giving up, "
                             "nothing picked")
                return None
            continue

        short = float(_LAST_LID_DIMS.get("short", cfg.BOX_LID_SHORT_FALLBACK_M))
        logger.info("paper lid: cup ({:.3f},{:+.3f}) at ee_z {:.4f}, detected size "
                    "{:.3f} x {:.3f} m", pose[0], pose[1], pose[2],
                    _LAST_LID_DIMS.get("long", 0.0), short)
        # If the centre will not seal, retry along the lid's SHORT axis (+y in
        # its own frame) — the same direction and distance the bin lid uses,
        # see BOX_LID_SEAL_RETRY_OFFSET_M for what this replaced.
        lyaw = float(np.deg2rad(det[2]))
        retry_dir = (-float(np.sin(lyaw)), float(np.cos(lyaw)))
        res = mover.pick(pose, expected_z=pose[2],
                         creep_gap=cfg.DESCENT_CREEP_GAP_M,
                         contact_n=float(cfg.BOX_LID_CONTACT_N),
                         force_limit=float(cfg.BOX_LID_FORCE_LIMIT_N),
                         retry_dir=retry_dir,
                         retry_offset_m=float(cfg.BOX_LID_SEAL_RETRY_OFFSET_M))
        if not res.success:
            if res.reason == "unreachable" and res.contact_ee_z is None:
                # hover leg failed before touching anything — nothing moved,
                # safe to nudge the chassis and redo detect + pick
                logger.warning("paper lid pick pose unreachable — adjust the chassis "
                               "(f/b/l/r), `d` to re-detect + retry, `q` to give up")
                if not _manual_strafe(bot, "adjust"):
                    logger.error("paper lid pick pose unreachable — giving up, "
                                 "nothing picked")
                    return None
                continue
            logger.error("paper lid pick failed: {}", res.reason)
            return None
        logger.info("paper lid picked (contact ee_z={:.4f})",
                    res.contact_ee_z if res.contact_ee_z is not None else float("nan"))
        return det, short


def run_box_lid(bot, mover: SuctionMover, gripper) -> bool:
    """--box-lid: take the PAPER box lid off with the cup, hand it to the RIGHT
    GRIPPER from the side, and drop it over the unload spot.

    Both chassis legs are hand-driven (`d` when in position, `q` gives up):

      1. drive to the box -> `d`
      2. detect the lid (box OBB class cfg.BOX_LID_CLS_ID, warped at
         cfg.BOX_LID_PLANE_Z_M, depth-refined like every lid detection), pick it
         at its CENTRE with the gentle cardboard force pair, lift to transport
      3. HANDOFF over the pick xy, lifted straight up to
         cfg.BOX_LID_HANDOFF_Z_M and then carried
         cfg.BOX_LID_HANDOFF_SHIFT_Y_M to +y — the smallest move that puts the
         lid's edge inside the gripper's reach, flown as a straight line at the
         creep speed because a sheet held by its centre on one cup peels off
         when hurried sideways. The
         gripper comes to the lid instead: its -y edge is short/2 along the LID's
         own -y axis (the detection measures short) and the WRIST SPINS to match
         the lid's yaw, so the fingers close parallel to that edge. What must
         not spin is the LEFT wrist — turning the sheet would sweep it through
         the space the gripper is entering.
      4. the gripper opens, starts cfg.BOX_LID_APPROACH_M out on the -y side
         (shortened when the arm cannot reach that far out) and enters in a
         STRAIGHT LINE (move_ee_line — a planned move would arc into the lid),
         cfg.BOX_LID_GRASP_DZ_M under the sheet, until the wrist feels
         cfg.BOX_LID_SIDE_CONTACT_N or it reaches cfg.BOX_LID_GRASP_INSET_M past
         the edge; then it closes. If it cannot reach the edge at all, the run
         stops with the lid still on the cup.
      5. suction blows off, the left arm lifts straight up and homes
      6. drive to the unload spot -> `d`
      7. torso leans to cfg.LID_PLACE_TORSO_DEG (the right arm cannot get below
         z 0.65 at the demo stance, which would drop the lid from 52cm), the
         gripper moves to cfg.BOX_LID_DROP_EE_POS and OPENS. No gentle place is
         wanted, so there is no descent-to-contact here at all.
      8. torso back to the demo stance

    Space is why the handoff is high: the lid comes all the way up to
    SAFE_TRANSPORT_Z before the gripper takes it.
    """
    if gripper is None:
        logger.error("--box-lid needs the right gripper (it is what carries the "
                     "lid) — none available")
        return False
    logger.info("position the chassis at the paper box, then `d`")
    if not _manual_strafe(bot, "start"):
        logger.error("start positioning aborted (`q`) — --box-lid cancelled")
        return False

    # --- 1) detect + pick at the CENTRE ---------------------------------
    got = _box_lid_pick(bot, mover)
    if got is None:
        return False
    det, short = got

    # --- 2) handoff: straight UP, then carry cfg.BOX_LID_HANDOFF_SHIFT_Y_M
    #        to +y ------------------------------------------------------------
    #     pick() already lifted to SAFE_TRANSPORT_Z over the pick xy, so the
    #     lift is normally a no-op. The carry that follows is what gives the
    #     gripper room to reach the lid's -y edge at all (see the config note) —
    #     it is a STRAIGHT line at the descent creep speed, the slowest leg
    #     here, because a big sheet held by its centre on one cup peels off when
    #     hurried sideways.
    pos_h, rpy_live = mover.current_ee_pose()
    rpy_hand = tuple(float(v) for v in rpy_live)
    if pos_h[2] < float(cfg.BOX_LID_HANDOFF_Z_M) - 1e-3:
        if mover.move_ee_vertical(float(cfg.BOX_LID_HANDOFF_Z_M), rpy_hand) is None:
            logger.error("could not lift to the handoff height — lid still HELD")
            return False
        pos_h, _ = mover.current_ee_pose()
    shift = float(cfg.BOX_LID_HANDOFF_SHIFT_Y_M)
    if shift > 1e-4:
        logger.info("carrying the lid {:.0f}mm to +y (y {:+.3f} -> {:+.3f}) so the "
                    "gripper can reach its edge", shift * 1000, float(pos_h[1]),
                    float(pos_h[1]) + shift)
        if mover.move_ee_line((float(pos_h[0]), float(pos_h[1]) + shift,
                               float(pos_h[2])), rpy_hand,
                              trace_tag="handoff carry") is None:
            logger.error("the carry stalled — lid still HELD, stopping here")
            return False
        pos_h, _ = mover.current_ee_pose()
    cx, cy = float(pos_h[0]), float(pos_h[1])
    logger.info("=== handoff: lid centre ({:.3f},{:+.3f}) yaw {:+.1f}deg, cup at "
                "z {:.3f} ===", cx, cy, det[2], float(pos_h[2]))

    # --- 3) the gripper takes it from the -y side -----------------------
    #     BOTH the edge offset and the WRIST SPIN follow the lid's own yaw: the
    #     fingers have to close parallel to the edge they are on. (Spinning the
    #     LEFT wrist to axis-align the sheet instead is what is forbidden — that
    #     sweeps 0.6m of paper through the space the gripper is entering.)
    #     Flown by hand 0906 11:55 (box_lid_jog, run 20260906_115137): fingers
    #     10mm under the sheet, in from 15mm past the edge, force stop 112mm past
    #     it, close — the lid came off the cup cleanly. cfg.BOX_LID_GRASP_INSET_M
    #     / _GRASP_DZ_M / _APPROACH_M are that run.
    edge, approach, _ = _lid_frame(pos_h, det[2], short)
    rpy_g = _side_rpy(det[2])
    ee_in = (np.array([edge[0], edge[1],
                       float(pos_h[2]) - float(cfg.SUCTION_LENGTH_M)
                       + float(cfg.BOX_LID_GRASP_DZ_M)])
             + approach * (float(cfg.BOX_LID_GRASP_INSET_M)
                           - float(cfg.BOX_FINGER_LENGTH_M)))
    # GripperMover IS the right arm's ArmMover (side="right",
    # ee_frame=GRIPPER_EE_FRAME) and it owns the right wrist's wrench — so it
    # flies the approach itself, and no second model gets built.
    right = gripper
    # The right arm's collision model holds the OTHER arm at the config it read
    # when it was BUILT — the left arm at home, before the pick. It is now up
    # holding the lid, so re-read it or every verdict below answers for an arm
    # that has moved.
    right._setup_model(None)
    right._setup_ik()
    right._setup_collision()

    # Where the entry STARTS: cfg.BOX_LID_APPROACH_M back from the aim, walked
    # in toward it in 20mm steps while that is out of reach. The gripper's -y
    # reach runs out around y -0.28 with the wrist pinned pointing +y, so the
    # far end of the entry is the part that fails first — offline at the nominal
    # (0.85,+0.05) with a 0.43m lid the full-length start is 23mm short while
    # the aim itself solves. A shorter entry is no loss; the force stop is what
    # ends it either way.
    def _reach(p) -> bool:                  # exactly the solve move_ee will do
        sol = right.solve_pose(tuple(p), rpy_g, seed=right._start_q(), min_motion=True)
        return (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits
                and not sol.in_collision)

    back = float(cfg.BOX_LID_APPROACH_M)
    while back > 1e-9 and not _reach(ee_in - approach * back):
        back -= 0.02
        logger.info("entry start {:.0f}mm out of the aim is unreachable — trying "
                    "{:.0f}mm", (back + 0.02) * 1000, back * 1000)
    if back <= 1e-9:
        logger.error("the gripper cannot reach the lid's edge where it was picked "
                     "(centre {:.3f},{:+.3f}, edge {:.3f},{:+.3f}) — lid still on "
                     "the cup, stopping", cx, cy, edge[0], edge[1])
        return False
    ee_hover = ee_in - approach * back
    logger.info("-y edge ({:.3f},{:+.3f}) -> fingertips aim {:.0f}mm past it, "
                "{:.0f}mm under the sheet: EE ({:.3f},{:+.3f},{:.3f}), entering "
                "from ({:.3f},{:+.3f}) {:.0f}mm out, wrist spun {:+.1f}deg",
                edge[0], edge[1], cfg.BOX_LID_GRASP_INSET_M * 1000,
                -cfg.BOX_LID_GRASP_DZ_M * 1000, *ee_in, ee_hover[0], ee_hover[1],
                back * 1000, det[2])
    gripper.gripper.open()
    if right.move_ee(tuple(ee_hover), rpy_g, quiet=False) is None:
        logger.error("the entry start went unreachable between the check and the "
                     "move — lid still on the cup, stopping")
        return False
    # Stop on FORCE, like the suction descent: the aim is deliberately deep and
    # contact is what ends the entry. Tared at the hover, where the arm is at
    # rest and nothing is touching.
    guard, seen = None, {"f": None}
    if right.tare_wrench():
        limit = float(cfg.BOX_LID_SIDE_CONTACT_N)

        def guard():                        # noqa: E306 — reads the approach axis
            f = right.axis_force(tuple(approach))
            seen["f"] = f
            return f is not None and f > limit
    else:
        logger.warning("right wrist not tared — side entry runs to its AIM with "
                       "no force stop")
    if right.move_ee_line(tuple(ee_in), rpy_g,
                          speed=float(cfg.BOX_LID_SIDE_SPEED_M_S),
                          stop_fn=guard, trace_tag="side entry") is None:
        logger.error("side entry stalled — retreating the gripper, lid still held")
        right.move_ee_line(tuple(ee_hover), rpy_g,
                           speed=float(cfg.BOX_LID_SIDE_SPEED_M_S))
        return False
    pos_in, _ = right.current_ee_pose()
    logger.info("entry ended {:.0f}mm past the edge ({:.0f}mm short of the aim), "
                "lateral force {}",
                (float(np.asarray(pos_in) @ approach)
                 - float(edge @ approach) + float(cfg.BOX_FINGER_LENGTH_M)) * 1000,
                float((ee_in - np.asarray(pos_in)) @ approach) * 1000,
                "n/a" if seen["f"] is None else "{:.1f}N".format(seen["f"]))
    y_shift = float(cfg.BOX_LID_GRASP_Y_SHIFT_M)
    if abs(y_shift) > 1e-9:
        grasp_pos = (float(pos_in[0]), float(pos_in[1]) + y_shift, float(pos_in[2]))
        logger.info("grasp y shift {:+.0f}mm ({:+.3f} -> {:+.3f}) before closing",
                    y_shift * 1000, float(pos_in[1]), grasp_pos[1])
        if right.move_ee(grasp_pos, rpy_g, quiet=False) is None:
            logger.error("grasp y-shift move failed — lid still on the cup, stopping")
            return False
    gripper.gripper.close()                 # blocks until the Robotiq reports done
    logger.info("gripper closed on the lid edge — releasing the cup")
    suction_io.release()
    time.sleep(float(cfg.CASE_PICK_RELEASE_WAIT_S))

    # From here the lid is held by the GRIPPER (not the cup) — a Ctrl+C
    # anywhere in this block should still open it before the interrupt
    # propagates, instead of leaving the lid clamped.
    try:
        # --- 4) the left arm gets out of the way ------------------------
        pos, _ = mover.current_ee_pose()
        mover.move_ee_vertical(min(float(cfg.SAFE_TRANSPORT_Z), float(pos[2]) + 0.05),
                               rpy_hand)
        mover.move_joints(mover._home_seed)

        # --- 5) drive to the unload spot, lean, drop --------------------
        logger.info("=== drive to the UNLOAD spot by hand — the GRIPPER holds "
                    "the lid ===")
        if not _manual_strafe(bot, "unload"):
            logger.error("unload positioning aborted (`q`) — opening the gripper")
            gripper.gripper.open()
            return False
        logger.warning("chassis at the unload spot — lid held in the gripper.")
        if input("Move torso now? [y/N]: ").strip().lower() != "y":
            logger.error("torso move declined — opening the gripper")
            gripper.gripper.open()
            return False
        torso_target = np.deg2rad(np.asarray(cfg.LID_PLACE_TORSO_DEG, dtype=float))
        logger.info("=== torso -> {} deg (the right arm holds the lid through "
                    "it) ===", cfg.LID_PLACE_TORSO_DEG)
        move_torso(bot.torso, torso_target, float(cfg.TORSO_VEL_SCALE), 60.0)
        right.pin_torso(torso_target)       # re-model both arms at the stance
        mover.pin_torso(torso_target)
        drop = tuple(float(v) for v in cfg.BOX_LID_DROP_EE_POS)
        logger.info("=== drop: gripper to ({:.3f},{:+.3f},{:.3f}) and OPEN ===",
                    *drop)
        if right.move_ee(drop, cfg.BOX_LID_SIDE_RPY, quiet=False) is None:
            logger.error("drop pose unreachable — lid still in the gripper, stopping")
            return False
    except KeyboardInterrupt:
        logger.error("interrupted (Ctrl+C) — opening the gripper before exiting")
        gripper.gripper.open()
        raise
    gripper.gripper.open()
    logger.info("=== --box-lid done: lid released from ee_z {:.3f} ===", drop[2])
    right.move_joints(right._home_seed)
    mover.pin_torso()                       # back to the demo stance
    return True


def run_lid_place(bot, mover: SuctionMover, yaw_delta: float = 0.0) -> bool:
    """Stack a HELD lid onto the lid lying on the floor: detect, resolve the cup
    xy / wrist yaw, descend to force and release, then restore the demo stance.

    Assumes the caller has already put the robot in the place stance
    (lid_place_stance) and that the lid is on the cup. Extracted from run_lid so
    ik_demo.lid_place can drive exactly this, unchanged — the place is the half
    that still fails, and a debugging tool with its own copy of it would drift
    from the one the run flies.

    ``yaw_delta``: the extra wrist rotation the PICK needed, carried here and
    re-added so the held lid still lands aligned with the floor lid (held lid yaw
    = wrist + (L_pick - W_pick) is invariant). 0.0 for a lid that was put on the
    cup by hand at the canonical yaw.

    A failed floor detection falls back to the taught cfg.LID_PLACE_XY_M rather
    than stranding the lid on the cup.
    """
    # --- Detect the lid ALREADY ON THE FLOOR and stack the held one onto it.
    #     Same geometry as the pick, read the other way round: the cup must end
    #     up over the same point of the lid it grabbed, so the grab offset is
    #     rotated by the TARGET lid's yaw and added to its center, and the wrist
    #     turns the held lid to that same yaw.
    set_head_pitch(bot, angle=cfg.LID_PLACE_HEAD_PITCH_DEG)
    # The config pair is the FALLBACK and the start-above-contact gap; the
    # measured surface replaces the contact height when depth answers.
    pz = float(cfg.LID_PLACE_EE_Z_M)
    sz = float(cfg.LID_PLACE_START_EE_Z_M)
    start_gap = sz - pz

    def _heights(t):
        """(start, contact) for a detection — measured if depth refined it."""
        if t is None or len(t) < 4 or t[3] is None:
            return float(cfg.LID_PLACE_START_EE_Z_M), float(cfg.LID_PLACE_EE_Z_M)
        c = float(t[3]) + float(cfg.SUCTION_LENGTH_M)
        logger.info("place heights from the DEPTH measurement: contact {:.4f}, "
                    "start {:.4f} (config had {:.3f} / {:.3f})", c, c + start_gap,
                    cfg.LID_PLACE_EE_Z_M, cfg.LID_PLACE_START_EE_Z_M)
        return c + start_gap, c
    # Detect first, and only move the chassis IF THE COLUMN DOES NOT SOLVE where
    # the lid already is. _center_lid drives to a taught reference xy and knows
    # nothing about reach, so calling it unconditionally re-parked the robot over
    # spots that were already fine (0904: a detection whose column solved 0.66
    # down to 0.28 still got corrected), and every open-loop correction adds its
    # own error. The alignment runs AFTER the lean either way, because the leaned
    # stance is what the column is judged in and the chassis is independent of
    # the torso; it also squares the lid, which is why a correction that DOES
    # run leaves the wrist at the canonical yaw.
    #
    # LOOPED (0906 10:25): one pass validates a spot at the xy and surface
    # height read BEFORE the chassis moves, then flies whatever it arrives at.
    # That run's 63mm alignment landed 6mm off in y (inside _center_lid's 20mm
    # deadband) and re-measured the surface 5.5mm lower, which together left the
    # bottom of the column 11.9mm short of REACH_TOL_M's 10mm — and with no
    # second round the lid stayed on the cup, when a few more cm of chassis
    # would have solved it. Re-asking after each arrival closes that gap; the
    # margin below makes the first answer robust enough that it rarely has to.
    tgt = _detect_lid_xy(bot, plane_z=cfg.LID_FLOOR_PLANE_Z_M)
    sz, pz = _heights(tgt)
    tries = int(cfg.LID_PLACE_ALIGN_MAX_TRIES)
    for attempt in range(1, tries + 1):
        if tgt is None or _place_column_ok(mover, tgt, yaw_delta, sz, pz):
            break
        spot = _nearest_place_spot(mover, tgt, yaw_delta, sz, pz,
                                   float(cfg.LID_PLACE_MIN_X_M),
                                   margin=float(cfg.LID_PLACE_SPOT_MARGIN_M))
        if spot is None:
            # Nothing with slack in hand. A bare-minimum spot still aims at
            # where the lid actually IS, which beats the taught reference — a
            # fixed point that ignores the detection entirely.
            spot = _nearest_place_spot(mover, tgt, yaw_delta, sz, pz,
                                       float(cfg.LID_PLACE_MIN_X_M))
            if spot is not None:
                logger.warning("no cup xy with {:.0f}mm of margin — taking the "
                               "nearest bare-minimum spot ({:.3f},{:+.3f})",
                               cfg.LID_PLACE_SPOT_MARGIN_M * 1000.0, *spot)
        if spot is None:
            spot = tuple(float(v) for v in cfg.LID_PLACE_XY_M)
            logger.warning("no solvable cup xy within reach of the lid — falling "
                           "back to the taught reference ({:.3f},{:+.3f})", *spot)
        logger.info("the place column does not solve where the lid is (try {}/{}) "
                    "— aligning the chassis to ({:.3f},{:+.3f})", attempt, tries, *spot)
        tgt = _center_lid(bot, plane_z=cfg.LID_FLOOR_PLANE_Z_M,
                          x_ref=spot[0], y_ref=spot[1],
                          x_min=float(cfg.LID_PLACE_MIN_X_M))
        sz, pz = _heights(tgt)          # a new arrival re-measures the surface
    else:
        logger.warning("the place column did not solve after {} alignment rounds "
                       "— the two-branch check below decides", tries)
    if tgt is None:
        px, py = (float(v) for v in cfg.LID_PLACE_XY_M)
        wyaw = float(cfg.GRASP_YAW) + yaw_delta
        logger.warning("floor lid NOT detected (class {}, plane z={:.3f}) — falling back "
                       "to the taught cup xy ({:.3f},{:+.3f})", cfg.LID_CLS_ID,
                       cfg.LID_FLOOR_PLANE_Z_M, px, py)
    else:
        # same helper the reachability test above used, so the checked aim and
        # the flown aim cannot drift apart
        px, py, wyaw = _lid_place_aim(tgt, yaw_delta)
        logger.info("floor lid: center ({:.3f},{:+.3f}) yaw {:+.1f}deg -> cup "
                    "({:.3f},{:+.3f}) at wrist yaw {:+.1f}deg (canonical {:+.1f} "
                    "{:+.1f}deg carried from the pick)", tgt[0], tgt[1], tgt[2],
                    px, py, float(np.rad2deg(wyaw)),
                    float(np.rad2deg(np.deg2rad(tgt[2]) + cfg.GRASP_YAW)),
                    float(np.rad2deg(yaw_delta)))
    # Cup back to VERTICAL — the torso move tilted it (roll/pitch ride the base)
    # and the descent has to come straight down. The yaw is the one computed
    # above, and ONLY that one. The wrist takes the detected lid's yaw, the same
    # rule the pick follows — see _pick_yaw_search for why the 180 flip is gone.
    # It bought nothing here anyway: the reachable wrist band at this leaned
    # stance excludes yaw near 0 deg, so the flipped branch of a lid detected
    # near the canonical yaw is hopeless by construction (0906 10:25 it missed
    # by 97.5mm while the real branch missed by 11.9mm), and a flip that DID
    # solve would not have fixed a mis-branched OBB: _lid_place_aim still
    # rotates LID_GRAB_OFFSET_M by the detected yaw, so the cup would aim at the
    # same point of the lid either way.
    _, held_rpy = mover.current_ee_pose()
    logger.info("cup re-levelled from roll/pitch {:+.1f}/{:+.1f}deg (the torso move "
                "tilted it), yaw was {:+.1f}deg", float(np.rad2deg(held_rpy[0])),
                float(np.rad2deg(held_rpy[1])), float(np.rad2deg(held_rpy[2])))
    cand = float((wyaw + np.pi) % (2.0 * np.pi) - np.pi)
    rpy = (float(cfg.GRASP_ORIENTATION_RPY[0]),
           float(cfg.GRASP_ORIENTATION_RPY[1]), cand)
    # NOT descent_reachable: its column runs down to DESCENT_CHECK_BOTTOM_EE_Z
    # (box floor + cup = 0.755), ABOVE this whole column — it would check a
    # single z that has nothing to do with the lid. Check the real one.
    if not mover.column_reachable(px, py, rpy, sz, pz):
        logger.error("lid place column ({:.3f} -> {:.3f}) out of reach at the "
                     "lid-aligned wrist {:+.1f}deg — lid still HELD, stopping here",
                     sz, pz, float(np.rad2deg(cand)))
        return False
    logger.info("lid place: xy=({:.3f},{:+.3f}) descend from ee_z {:.4f} to "
                "expected contact {:.4f} at wrist yaw {:+.1f}deg",
                px, py, sz, pz, float(np.rad2deg(cand)))
    place_pose = (px, py, pz, *rpy)
    # committed: pose resolved, column checked. Freeze the evidence.
    save_lid_frame(bot, "lid_place", lid_xy=None if tgt is None else tgt[:2],
                   cup_xy=(px, py), yaw_deg=None if tgt is None else tgt[2])
    pres = mover.place(place_pose, expected_z=pz, approach_z=sz, lift_z=sz,
                       creep_gap=cfg.DESCENT_CREEP_GAP_M)   # stacked-lid height unmeasured
    if not pres.success:
        # place() already released at the operator gate (Enter) and retreated
        # to the approach height, so the lid is down and the cup is clear:
        # carry on to the stance restore rather than leaving the robot leaned
        # over with the arm out.
        logger.warning("lid place failed ({}) — released at the gate, restoring the "
                       "stance anyway", pres.reason)
    else:
        logger.info("=== --lid done: lid released at ee_z={:.4f} ===",
                    pres.contact_ee_z if pres.contact_ee_z is not None else float("nan"))
    # Back to the taught stance, and BOTH arms back to home, TOGETHER — the
    # mirror image of lid_place_stance and for the same reason: the torso
    # carries both arm bases, so un-leaning it with the arms frozen drags them
    # through whatever pose that combination produces. Safe to move the arm
    # here because the lid is already released and place() has retreated to its
    # approach height, so the cup is clear of what it just put down.
    logger.info("=== torso -> demo stance {} AND both arms -> home, together ===",
                np.round(cfg.TORSO_JOINTS, 4))
    right = _right_arm(bot)
    _park_during_legs("lid",
                      [lambda: mover.move_joints(mover._home_seed),
                       lambda: right.move_joints(right._home_seed)],
                      lambda: move_torso(bot.torso,
                                         np.asarray(cfg.TORSO_JOINTS, dtype=float),
                                         float(cfg.LID_TORSO_VEL_SCALE), 60.0))
    mover.pin_torso()          # arrives already there: re-models only
    return bool(pres.success)


def run_lid(bot, mover: SuctionMover) -> bool:
    """--lid: take the LID off a box with the suction cup, then stack it on the
    lid already lying on the floor at the unload spot.

    Both chassis legs are hand-driven — the operator has the keyboard, `d` when
    in position (`q` gives up):

      1. drive to the lidded box -> `d`
      2. BEV-detect the lid (class cfg.LID_CLS_ID on the 2-class bin model,
         warped at cfg.LID_PLANE_Z_M), pick it at cfg.LID_GRAB_OFFSET_M from
         its center in the LID's own frame, lift, park out of camera view
      3. `d` (after backing off the box if needed): torso down to
         cfg.LID_PLACE_TORSO_DEG (the floor column does not solve at the demo
         stance) and both arms to their stow joints, together
      4. drive to the unload spot in that stance, lid on the cup -> `d`;
         head to cfg.LID_PLACE_HEAD_PITCH_DEG
      5. detect the lid ON THE FLOOR (same class, plane
         cfg.LID_FLOOR_PLANE_Z_M) and stack the held one onto it: the cup goes
         over the same point of the lid it grabbed, the wrist turns the held lid
         to the detected yaw, and the descent stops on force
      6. torso back to the demo stance

    A failed floor detection falls back to the taught cfg.LID_PLACE_XY_M rather
    than stranding the lid on the cup.
    """
    logger.info("position the chassis at the lidded box, then `d`")
    if not _manual_strafe(bot, "start"):
        logger.error("start positioning aborted (`q`) — --lid cancelled")
        return False

    # Detect, try to solve the pick WHERE THE LID IS, and only align the
    # chassis if nothing solves — same policy as the place side. _center_lid
    # drives to a taught reference and knows nothing about reach, so calling it
    # unconditionally re-parks the robot over spots that were already fine, and
    # every open-loop correction adds its own error.
    det = _detect_lid_xy(bot)
    pose, rpy, yaw_delta = ((None, None, 0.0) if det is None
                            else _pick_yaw_search(mover, det))
    if det is not None:
        gx, gy, ez = _lid_pick_aim(det)
        logger.info("lid grab: center ({:.3f},{:+.3f}) yaw {:+.1f}deg -> cup "
                    "({:.3f},{:+.3f}) at ee_z {:.4f} (offset {:+.0f},{:+.0f}mm in "
                    "the lid frame)", det[0], det[1], det[2], gx, gy, ez,
                    cfg.LID_GRAB_OFFSET_M[0] * 1000, cfg.LID_GRAB_OFFSET_M[1] * 1000)
    # Reach correction: the chassis goes to the NEAREST cup xy where the pick
    # column solves (_nearest_pick_spot, with slack; bare pass second; the
    # taught LID_CENTER_XY_M only if nothing within the clamp solves at all),
    # re-detects, re-solves — up to CHASSIS_ADJUST_MAX_ATTEMPTS rounds, since
    # one open-loop move lands 1-2 cm off what it aimed at.
    adjusts = 0
    while det is not None and pose is None and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
        adjusts += 1
        spot = _nearest_pick_spot(mover, det, margin=float(cfg.CHASSIS_ADJUST_REACH_MARGIN_M))
        if spot is None:
            spot = _nearest_pick_spot(mover, det)
        if spot is None:
            spot = tuple(float(v) for v in cfg.LID_CENTER_XY_M)
            logger.warning("no solvable cup xy within {:.2f} m of the lid — falling back to "
                           "the taught reference ({:.3f},{:+.3f})",
                           cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M, *spot)
        logger.info("no wrist yaw solves the pick column where the lid is (try {}/{}) — "
                    "aligning the chassis so the cup point sits at ({:.3f},{:+.3f})",
                    adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS, *spot)
        det = _center_lid(bot, x_ref=spot[0], y_ref=spot[1],
                          x_min=float(cfg.LID_CENTER_MIN_X_M))
        if det is not None:
            pose, rpy, yaw_delta = _pick_yaw_search(mover, det)
    if det is None:
        logger.error("lid NOT detected / rejected (class {} on the bin OBB model, warp "
                     "plane z={:.3f}) — nothing to pick", cfg.LID_CLS_ID, cfg.LID_PLANE_Z_M)
        return False
    if pose is None:
        logger.error("no wrist yaw solves the lid pick column, even after {} alignment "
                     "round(s) — not moving", adjusts)
        return False
    # No creep_gap override: pick() now descends per-tick from the hover and
    # runs through the creep line without stopping, so its default gap is the
    # long DESCENT_CREEP_GAP_M (0909). ez is the depth-MEASURED surface (see
    # _lid_pick_aim); 0905 15:28 measured 17mm of error in it (predicted contact
    # 0.9574, cup touched at 0.940), which the 50mm gap covers.
    # retry_dir is the LID's own +y, the same frame LID_GRAB_OFFSET_M lives in,
    # so a seal retry samples a fresh patch of the lid instead of re-pressing the
    # crease or embossing step that just refused to seal.
    lid_yaw = float(np.deg2rad(det[2]))
    res = mover.pick(pose, expected_z=ez,
                     retry_dir=(-float(np.sin(lid_yaw)), float(np.cos(lid_yaw))))
    if not res.success:
        logger.error("lid pick failed: {} — chassis stays put", res.reason)
        return False
    logger.info("lid picked (contact ee_z={:.4f}) — holding it on the cup",
                res.contact_ee_z if res.contact_ee_z is not None else float("nan"))

    # NO view park after the pick. Nothing detects before the unload station,
    # and a park LOWERS the box clearance rather than raising it: the lift
    # already ends at SAFE_TRANSPORT_Z, which puts the lid 55mm over the box it
    # came off, while a park at 1.08 leaves 35mm. The arm holds where the lift
    # left it until the operator triggers the place stance — one less lateral
    # sweep, and one less failure mode (a park unreachable at the held yaw used
    # to fall back to the joint park, cup pitched, with the lid on it).
    #
    # The stance change (torso lean + both arms to stow, see lid_place_stance)
    # is its own `d`-triggered step BEFORE the unload drive, so the operator can
    # back the chassis off the box first and the drive happens with the arm
    # already tucked, not swinging out at transport height.
    logger.info("=== lid on the cup — back off the box if needed, then `d` = "
                "torso lean + arms to stow ===")
    if not _manual_strafe(bot, "stance"):
        logger.error("stance step aborted (`q`) — lid still HELD, stopping here")
        return False
    lid_place_stance(bot, mover)

    logger.info("=== drive to the UNLOAD spot by hand — the lid stays on the cup ===")
    if not _manual_strafe(bot, "unload"):
        logger.error("unload positioning aborted (`q`) — lid still HELD, stopping here")
        return False

    return run_lid_place(bot, mover, yaw_delta)


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
        xy = None if rgb is None else dbn.find_bin_bev(
            rgb, *_joints(bot), plane_z=cfg.DIVERT_BIN_PLANE_Z_M,
            cls_id=cfg.BIN_CLS_ID)
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


def _divert_case_place(bot, mover: SuctionMover, label: str, slot_key: str,
                       auto: bool, leg: "ChassisNav | None",
                       divert_leg: "ChassisNav | None",
                       zt: "ZTracker | None") -> "bool | None":
    """Suction-place the held TARGET battery into the DIVERT CASE to the LEFT
    of the source: strafe left (open-loop, view park in parallel) by
    `divert_leg`'s LEARNED distance (own ChassisNav instance, seeded from
    DIVERT_CASE_STRAFE_LEFT_M — a different physical gap than the main
    source<->target leg, so it must not teach or be taught by `leg`),
    BEV-detect the divert case (centered like a normal place under
    --auto-move; its arrival residual DOES teach divert_leg), place at
    `slot_key` (BAT_SRC_2 = left slot / BAT_SRC_1 = right slot of the
    DETECTED case) with the SAME logic as the target battery place — yaw
    trim, corner-seat aim bias, reach pre-check, re-detect / auto-adjust /
    keyboard ladder, corner seat on contact — then strafe back right to the
    source (view park in parallel) by the same divert_leg distance. The excursion is
    unattributable to the MAIN leg, so leg.skip_next_learn is set for the
    next centering.

    Returns True once the battery is in the divert case (robot back at the
    source); False with the battery STILL ON THE CUP after a detect/reach miss
    (robot back at the source) so the caller can seat it in the target case
    instead; None if the place failed with the part still held — the run must
    stop (robot left at the divert case)."""
    from .move_chassis import strafe_left, strafe_right
    dist = divert_leg.dist("left") if divert_leg is not None else cfg.DIVERT_CASE_STRAFE_LEFT_M
    logger.info("[{}] divert: strafe LEFT {:.2f} m to the divert case (slot {})",
                label, dist, slot_key)
    # CHASSIS_LEG_SPEED_MS like the source<->target legs (strafe()): these are
    # long station legs too, not corrections — the centering / adjust moves at
    # the divert case stay at the slow default
    _park_during_legs(label, [lambda: _view_park(mover, label)],
                      lambda: strafe_left(bot, distance_m=dist,
                                          speed=cfg.CHASSIS_LEG_SPEED_MS))
    if leg is not None:
        leg.skip_next_learn = True
    # slot point on the center line, same per-item ref scheme as run_item
    y_ref = cfg.CHASSIS_CENTER_CASE_Y_M - resolve_poses((0.0, 0.0, 0.0, 0.0))[slot_key][1]
    # Same ladder as run_item's battery place: re-detect once on a miss, then
    # auto-adjust (nearest reachable spot) up to CHASSIS_ADJUST_MAX_ATTEMPTS on
    # an out-of-reach pose, then the keyboard. `q` there does not stop the run:
    # it gives the divert up and the battery is seated in the target case.
    place_pose = None
    det = None
    redetects = 0
    adjusts = 0
    while True:
        det = (_center_case(bot, cfg.DIVERT_CASE_LAYERS, label, "divert", divert_leg, y_ref)
               if auto else detect(bot, cfg.DIVERT_CASE_LAYERS))
        if det is None or not det.found:
            if redetects < 1:
                redetects += 1
                logger.warning("[{}] divert case NOT detected — re-detecting once", label)
                continue
            logger.warning("[{}] divert case NOT detected — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to give up the divert (the "
                           "battery goes to the target case)", label)
            if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
                break
            continue
        det = _refine_det(bot, cfg.DIVERT_CASE_LAYERS, det)   # median-of-N for the pose
        _log_det(zt, "divert", label, cfg.DIVERT_CASE_LAYERS, det)
        logger.info("[{}] divert case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg conf={:.2f}",
                    label, det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.conf)
        pose = resolve_poses(_center_from_det(det))[slot_key]
        if cfg.PLACE_YAW_TRIM_RAD:
            # same systematic in-hand twist as the target battery place
            pose = (*pose[:5], pose[5] + cfg.PLACE_YAW_TRIM_RAD)
        # corner-seat aim bias, as the target battery place — applied BEFORE the
        # reach pre-check so the checked pose is the flown one
        bias_x, bias_y = cfg.BATTERY_CORNER_AIM_BIAS_M
        pose = (pose[0] - np.sign(cfg.CASE_CORNER_DIR[0]) * bias_x,
                pose[1] - np.sign(cfg.CASE_CORNER_DIR[1]) * bias_y, *pose[2:])
        logger.info("[{}] corner-seat aim bias ({:+.1f},{:+.1f})mm (away from the "
                    "datum corner)", label,
                    -np.sign(cfg.CASE_CORNER_DIR[0]) * bias_x * 1000,
                    -np.sign(cfg.CASE_CORNER_DIR[1]) * bias_y * 1000)
        if descent_reachable(mover, pose):
            place_pose = pose
            break
        if auto and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
            adjusts += 1
            logger.warning("[{}] divert slot pose out of reach — auto-adjust {}/{}",
                           label, adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
            dy = _auto_adjust(bot, mover, det, pose)
            if dy is not None:
                if divert_leg is not None:
                    divert_leg.learn(dy, "divert")
                continue
            # nothing within the adjust clamp solves -> straight to the keyboard
        logger.warning("[{}] divert slot pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to give up the divert (the battery "
                       "goes to the target case)", label)
        if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
            break
    if place_pose is None:
        if zt is not None:
            zt.log_event("divert_miss", "divert", label, cfg.DIVERT_CASE_LAYERS)
        logger.warning("[{}] divert aborted — strafe back right to the source", label)
        _park_during_legs(label, [lambda: _view_park(mover, label)],
                          lambda: strafe_right(bot, distance_m=dist,
                                               speed=cfg.CHASSIS_LEG_SPEED_MS))
        return False
    # no z anchor for this station — expectation from the detected plane, with
    # the loose battery-over-case tolerance (seat at most one battery thickness
    # above the case face)
    exp_z = det.top_face_z + cfg.SUCTION_LENGTH_M
    # corner_seat="battery": driven into the slot corner after contact, exactly
    # as the same battery would be in the target case
    pres = mover.place(place_pose, expected_z=exp_z,
                       misseat_tol_m=cfg.BATTERY_OVER_CASE_MAX_M,
                       lift_to_clear=True, corner_seat="battery")
    if pres is not None and not getattr(pres, "success", True):
        if zt is not None:
            zt.log_event("place_" + pres.reason, "divert", label,
                         cfg.DIVERT_CASE_LAYERS, pres.contact_ee_z, exp_z)
        if pres.reason == "unreachable" and pres.contact_ee_z is None:
            # hover leg failed BEFORE the release gate — part still on the cup
            logger.error("[{}] divert place failed: {} (part still held) — stopping "
                         "at the divert case", label, pres.reason)
            return None
        logger.warning("[{}] divert place failed ({}) — operator resolved it at the "
                       "gate, continuing", label, pres.reason)
    logger.info("[{}] diverted to the case slot {} — strafe back right to the source",
                label, slot_key)
    _park_during_legs(label, [lambda: _view_park(mover, label)],
                      lambda: strafe_right(bot, distance_m=dist,
                                           speed=cfg.CHASSIS_LEG_SPEED_MS))
    return True


def _place_case_in_bin(bot, mover: SuctionMover, label: str, source_yaw: float,
                       layers_in_bin: int, auto: bool):
    """Place the HELD case into the bin box from a BIN detection — NOT from a
    detection of whatever case may already be in it. Center = _detect_bin_xy
    + BIN_PLACE_CENTER_OFFSET (its OWN offset: the seed's SEED_BIN_CENTER_OFFSET
    is paired with the corner-seat aim bias + drive, which this plain place
    does not have — see config); yaw = the yaw the case was PICKED at (no
    de-rotation in transit); seat z modelled from the bin floor with
    `layers_in_bin` cases already in it (0 = empty bin -> the seed place's
    TARGET_DEFAULT_CASE_CENTER height). Plain aligned place, no corner seat
    (the bin's wall geometry is not the target jig's); expected_z stays None so
    place() gates the misseat off that modelled seat z (SEED_MISSEAT_TOL_M,
    25mm: a rim/wall landing contacts well above it and is HELD, a one-layer
    count error still releases).

    A missed detection first walks SEED_BIN_SEARCH_STRAFES_M when `auto`
    (changes the view), then hands the operator the keyboard (`d` re-detects
    from the new base frame, `q` stops). An out-of-reach pose first gets
    _auto_adjust (nearest reachable spot, CHASSIS_ADJUST_MAX_ATTEMPTS rounds)
    in EVERY mode — `auto` only gates the search strafes — then the same prompt; a hover leg that fails BEFORE the
    release gate goes to the prompt too — the case is still on the cup in all
    three. Returns the PlaceResult once a place ran (it may
    still report a failure that released at the gate), or None if the operator
    stopped with the case held."""
    from .move_chassis import strafe_left, strafe_right
    seat_z = (cfg.FLOOR_Z_BASE_M + (int(layers_in_bin) + 1) * cfg.LAYER_PITCH_M
              + cfg.SUCTION_LENGTH_M)
    search = [float(s) for s in cfg.SEED_BIN_SEARCH_STRAFES_M] if auto else []
    adjusts = 0
    while True:
        bxy = _detect_bin_xy(bot)
        if bxy is None:
            logger.warning("[{}] bin NOT detected", label)
            if search:
                move = search.pop(0)
                logger.warning("[{}] bin search: strafe {} {:.2f} m, re-detect "
                               "({} step(s) left)", label,
                               "LEFT" if move > 0 else "RIGHT", abs(move), len(search))
                (strafe_left if move > 0 else strafe_right)(bot, distance_m=abs(move))
                continue
            logger.warning("[{}] adjust the chassis (f/b/l/r), `d` to re-detect, "
                           "`q` to stop (case still held)", label)
            if not _manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the bin (case still held)", label)
                return None
            continue
        center = (bxy[0] + cfg.BIN_PLACE_CENTER_OFFSET[0],
                  bxy[1] + cfg.BIN_PLACE_CENTER_OFFSET[1], seat_z, float(source_yaw))
        logger.info("[{}] bin center ({:.3f},{:+.3f}) -> place ({:.3f},{:+.3f}) seat "
                    "ee_z {:.3f} ({} case(s) already in the bin), at the pick yaw "
                    "{:+.1f} deg", label, bxy[0], bxy[1], center[0], center[1], seat_z,
                    int(layers_in_bin), float(np.rad2deg(source_yaw)))
        place_pose = resolve_poses(center)["CASE_PICK"]
        if not descent_reachable(mover, place_pose):
            if adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:      # reach fix: every mode
                adjusts += 1
                logger.warning("[{}] bin place pose out of reach — auto-adjust {}/{}",
                               label, adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
                if _auto_adjust(bot, mover, None, place_pose) is not None:
                    continue                # re-detect the bin from the new frame
            logger.warning("[{}] bin place pose out of reach — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to stop (case still held)", label)
            if not _manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the bin (case still held)", label)
                return None
            continue
        pres = mover.place(place_pose, misseat_tol_m=cfg.SEED_MISSEAT_TOL_M,
                           lift_to_clear=True)
        if (pres is not None and not getattr(pres, "success", True)
                and pres.reason == "unreachable" and pres.contact_ee_z is None):
            # hover leg failed BEFORE the release gate — case still on the cup,
            # nothing moved: reposition, `d` re-detects the bin from the new frame
            logger.warning("[{}] bin place unreachable (case still held) — adjust the "
                           "chassis (f/b/l/r), `d` to re-detect + retry, `q` to stop", label)
            if _manual_strafe(bot, "adjust"):
                continue
            logger.error("[{}] stopping at the bin (case still held)", label)
            return None
        return pres


def _final_case_to_bin(bot, mover: SuctionMover, auto: bool,
                       leg: "ChassisNav | None",
                       zt: "ZTracker | None") -> bool:
    """run() epilogue (cfg.FINAL_CASE_TO_BIN): ONE case is still in the source
    box after the layer loop — pick it (source station, layers=1), strafe
    RIGHT the fixed FINAL_CASE_STRAFE_RIGHT_M leg to the bin box (view park in
    parallel), align to the detected bin, then place from the BIN detection
    (_place_case_in_bin — the case already in the bin is not detected, its
    count FINAL_BIN_CASE_LAYERS only sets the modelled seat z), then strafe
    back LEFT the same leg and finish. Both legs are fixed excursions, so ChassisNav learning is skipped
    (as in the divert). Returns False with the robot left where the failure
    happened (pick fail: source, arms homed; place fail: bin side)."""
    label = "final_case"
    from .move_chassis import strafe_left, strafe_right
    y_ref = cfg.CHASSIS_CENTER_CASE_Y_M - resolve_poses((0.0, 0.0, 0.0, 0.0))["CASE_PICK"][1]

    # --- SOURCE: detect the last case + pick (run_item's source flow at layers=1)
    src_plane = zt.plane_z("source", 1) if zt is not None else None
    adjusts = 0
    while True:
        det = (_center_case(bot, 1, label, "source", leg, y_ref, src_plane,
                            x_ref=cfg.SOURCE_CASE_CENTER[0])
               if auto else detect(bot, 1, src_plane))
        if det is None or not det.found:
            logger.error("[{}] final source case NOT detected — stopping at the source", label)
            return False
        det = _refine_det(bot, 1, det, src_plane)
        _log_det(zt, "source", label, 1, det)
        logger.info("[{}] final source case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg conf={:.2f}",
                    label, det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.conf)
        center = _center_from_det(det)
        pick_pose = resolve_poses(center)["CASE_PICK"]
        if descent_reachable(mover, pick_pose):
            break
        if auto and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
            adjusts += 1
            logger.warning("[{}] pick pose out of reach — auto-adjust {}/{}",
                           label, adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
            dy = _auto_adjust(bot, mover, det, pick_pose)
            if dy is not None:
                if leg is not None:
                    leg.learn(dy, "source")
                continue
            # nothing within the adjust clamp solves -> straight to the keyboard
        logger.warning("[{}] pick pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to give up", label)
        if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
            logger.error("[{}] giving up the final case (robot left at the source)", label)
            return False
    pz = zt.expected_ee_z("source", "case", 1) if zt is not None else None
    exp_z = pz if pz is not None else det.top_face_z + cfg.SUCTION_LENGTH_M
    # creep out of the bin before cruising: the case lifts from between the
    # source bin walls, where a fast start throws the cup tens of mm off line
    res = mover.pick(pick_pose, expected_z=exp_z, lift_to_clear=True,
                     lift_creep_out_m=cfg.DESCENT_CREEP_GAP_M)
    if not res.success:
        logger.error("[{}] final pick failed: {} — arms home, stopping at the source",
                     label, res.reason)
        if zt is not None:
            zt.log_event("pick_" + res.reason, "source", label, 1, res.contact_ee_z, exp_z)
        _arms_home(bot, mover)
        return False
    if zt is not None:
        zt.record("source", label, 1, res.contact_ee_z)

    # --- RIGHT leg to the bin box (view park in parallel), fixed distance
    logger.info("[{}] strafe RIGHT {:.2f} m to the bin box", label,
                cfg.FINAL_CASE_STRAFE_RIGHT_M)
    _park_during_legs(label, [lambda: _view_park(mover, label)],
                      lambda: strafe_right(bot, distance_m=cfg.FINAL_CASE_STRAFE_RIGHT_M,
                                           speed=cfg.CHASSIS_LEG_SPEED_MS))
    if leg is not None:
        leg.skip_next_learn = True

    # --- BIN: closed-loop bin align (absorbs the long leg's open-loop drift;
    #     the y gate rejects the TARGET box one leg to the left), then place
    #     from the BIN detection (see _place_case_in_bin). The case already in
    #     the bin is NOT detected — its count only sets the modelled seat z.
    _align_to_bin(bot, label, fallback_right_m=0.0,
                  max_err_m=cfg.CHASSIS_DETECT_Y_GATE_M)
    layers = int(cfg.FINAL_BIN_CASE_LAYERS) + 1
    pres = _place_case_in_bin(bot, mover, label, center[3],
                              int(cfg.FINAL_BIN_CASE_LAYERS), auto)
    if pres is None:
        return False                        # operator stopped, case still held
    if not getattr(pres, "success", True):
        if zt is not None:
            zt.log_event("place_" + pres.reason, "bin", label, layers, pres.contact_ee_z)
        logger.warning("[{}] bin place failed ({}) — operator resolved it at the "
                       "gate, continuing", label, pres.reason)
    elif zt is not None:
        zt.record("bin", label, layers, pres.contact_ee_z)

    # --- return leg: back LEFT to the start side, view park in parallel
    logger.info("[{}] case on the bin stack — strafe back LEFT {:.2f} m", label,
                cfg.FINAL_CASE_STRAFE_RIGHT_M)
    _park_during_legs(label, [lambda: _view_park(mover, label)],
                      lambda: strafe_left(bot, distance_m=cfg.FINAL_CASE_STRAFE_RIGHT_M,
                                          speed=cfg.CHASSIS_LEG_SPEED_MS))
    return True


def run_item(bot, mover: SuctionMover, label: str, pose_key: str,
             src_layers: int, tgt_layers: int, next_pose_key: str,
             scan: bool = False, auto: bool = False,
             leg: "ChassisNav | None" = None,
             divert_leg: "ChassisNav | None" = None,
             zt: "ZTracker | None" = None,
             divert_slots: "list[str] | None" = None,
             strict_start: bool = False) -> bool:
    """One item's full left->pick->right->place->left cycle.

    `src_layers` / `tgt_layers` are the CURRENT stack heights for this layer
    (the layer loop in run() steps them; cfg.SRC/TGT_LAYERS_REMAINING are only
    the starting values). `next_pose_key` is the item that will be picked
    after this one returns to source (case/battery_1/battery_2 cycle) — its
    centering ref is KNOWN geometry, so the return LEFT leg folds that
    deliberate offset into the same open-loop move (ChassisNav.ref_change)
    instead of a separate corrective strafe; the detection-based centering at
    the next item still runs as the residual safety net. With `scan`, battery picks read the barcode during
    the descent; a TARGET_BARCODES match is suction-placed into the DIVERT
    CASE to the LEFT of the source (_divert_case_place, using `divert_leg`'s
    own learned distance — a separate ChassisNav instance from `leg`, since
    it's a different physical gap) — `divert_slots` is run()'s remaining
    slot-key order (left slot first, then right; the used slot is consumed
    on success). With `auto` (--auto-move), chassis legs are
    automatic and a failed reach pre-check auto-adjusts from the detection (up
    to CHASSIS_ADJUST_MAX_ATTEMPTS) before falling back to the interactive
    keyboard prompt. Every TARGET place corner-seats (suction.place
    (corner_seat=)): the descent aims off the datum corner and the held part
    is driven into it, each axis stopping on its own wall contact, so the
    walls fix its final position — the case drives WHILE descending (tall
    bin walls), the battery only AFTER contact (low slot walls)."""
    logger.info("=== item: {} ({}) src_layers={} tgt_layers={} ===",
                label, pose_key, src_layers, tgt_layers)
    # Per-item centering ref: put THIS item's grab/seat point (its case-local
    # y offset at yaw 0) on the center line, not the case center.
    y_ref = cfg.CHASSIS_CENTER_CASE_Y_M - resolve_poses((0.0, 0.0, 0.0, 0.0))[pose_key][1]
    # NEXT item's ref, for the return LEFT leg's combined move (see docstring).
    next_y_ref = (cfg.CHASSIS_CENTER_CASE_Y_M
                  - resolve_poses((0.0, 0.0, 0.0, 0.0))[next_pose_key][1])

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
        # The run's FIRST source centering follows the operator's manual park:
        # tighter deadbands and more rounds, since every later leg inherits it
        det = (_center_case(bot, src_layers, label, "source", leg, y_ref, src_plane,
                            tol_m=(cfg.CHASSIS_START_CENTER_TOL_M
                                   if strict_start else None),
                            min_turn_deg=(cfg.CHASSIS_START_MIN_TURN_DEG
                                          if strict_start else None),
                            max_moves=(cfg.CHASSIS_START_CENTER_MAX_MOVES
                                       if strict_start else None),
                            x_ref=cfg.SOURCE_CASE_CENTER[0])
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
            dy = _auto_adjust(bot, mover, det, pick_pose)
            if dy is not None:
                if leg is not None:
                    leg.learn(dy, "source")
                continue
            # nothing within the adjust clamp solves -> straight to the keyboard
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
    pz = zt.expected_ee_z("source", label, src_layers) if zt is not None else None
    exp_z = pz if pz is not None else det.top_face_z + cfg.SUCTION_LENGTH_M
    if label.startswith("battery"):
        _log_pick_depth(bot, label, pick_pose, src_plane, src_layers, exp_z, zt)
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
        # only the CASE creeps out of its lift: it comes up from between the
        # source bin walls. A battery leaves its compartment clear, and a place
        # lifts an empty cup, so both take the fast profile (user, 0903).
        res = mover.pick(pick_pose, expected_z=exp_z, lift_to_clear=True,
                         lift_creep_out_m=(cfg.DESCENT_CREEP_GAP_M
                                           if label == "case" else 0.0))
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

    # --- DIVERT (barcode-matched battery): suction-place into the divert case
    #     to the LEFT of the source, remembered slot order (left, then right);
    #     a miss falls through to the normal target-case place below ---
    if is_target(res.barcode):
        if divert_slots:
            logger.info("[{}] TARGET battery {!r} — diverting to the divert case",
                        label, res.barcode)
            diverted = _divert_case_place(bot, mover, label, divert_slots[0],
                                          auto, leg, divert_leg, zt)
            if diverted is None:
                # part still on the cup at the divert case — stop here
                logger.error("[{}] stopping at the divert case (part still held)", label)
                return False
            if diverted:
                divert_slots.pop(0)
                return True
            logger.warning("[{}] divert aborted — seating in the target case instead",
                           label)
        else:
            logger.warning("[{}] TARGET battery {!r} but both divert slots are used — "
                           "seating in the target case", label, res.barcode)

    # --- TARGET: park the arm (clear the head view) IN PARALLEL with the
    #     strafe right, then detect + place in the case ---
    _park_during_legs(label, [lambda: _view_park(mover, label)],
                      lambda: strafe(bot, "right", auto, leg))
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
            tdet = (_center_case(bot, tgt_layers, label, "target", leg, y_ref, tgt_plane,
                                 x_ref=cfg.TARGET_DEFAULT_CASE_CENTER[0])
                    if auto else detect(bot, tgt_layers, tgt_plane))

        if seed:
            logger.info("[{}] empty target (first case) -> seed place, no case "
                        "detection", label)
            from .move_chassis import strafe_left, strafe_right
            if auto:
                # Strafe so the BIN center sits where the seed will land the
                # case center (TARGET_DEFAULT y) — the closed-loop detection
                # below then only absorbs the small aligned residual. This IS
                # the right leg's arrival measurement (no separate case
                # detection runs for the seed), so its residual — net of the
                # deliberate ref change from this item's y_ref to the bin's
                # target_y — teaches the shared leg distance directly.
                deliberate = leg.ref_change(cfg.TARGET_DEFAULT_CASE_CENTER[1]) if leg is not None else 0.0
                dy = _align_to_bin(bot, label,
                                   target_y=cfg.TARGET_DEFAULT_CASE_CENTER[1],
                                   fallback_right_m=0.0,
                                   max_err_m=cfg.CHASSIS_DETECT_Y_GATE_M)
                if leg is not None and dy != 0.0:
                    # dy==0.0 also covers "no bin found, stayed put" (no real
                    # measurement) — only teach on an actual observed move.
                    leg.learn(dy - deliberate, "target")
            # CLOSED-LOOP seed: the place center comes from a bin detection
            # (+ SEED_BIN_CENTER_OFFSET), trusted AS-IS (no deviation gates —
            # a bad aim is still caught by the reach pre-check and by the
            # corner-seat's wall-latch release precondition). Detection is
            # REQUIRED — there is no blind default-pose place anymore: a
            # failed detection first walks SEED_BIN_SEARCH_STRAFES_M (auto)
            # to change the view, then hands the operator the keyboard (move
            # the chassis, `d` re-detects, `q` stops with the case held).
            search = [float(s) for s in cfg.SEED_BIN_SEARCH_STRAFES_M]
            while True:
                bxy = _detect_bin_xy(bot)
                if bxy is not None:
                    seed_xy = (bxy[0] + cfg.SEED_BIN_CENTER_OFFSET[0],
                               bxy[1] + cfg.SEED_BIN_CENTER_OFFSET[1])
                    logger.info("[{}] seed place from the detected bin: center "
                                "({:.3f},{:+.3f}) -> place ({:.3f},{:+.3f})", label,
                                bxy[0], bxy[1], seed_xy[0], seed_xy[1])
                    break
                logger.warning("[{}] seed bin detect failed", label)
                if auto and search:
                    move = search.pop(0)
                    logger.warning("[{}] seed bin search: strafe {} {:.2f} m, "
                                   "re-detect ({} step(s) left)", label,
                                   "LEFT" if move > 0 else "RIGHT", abs(move),
                                   len(search))
                    (strafe_left if move > 0 else strafe_right)(
                        bot, distance_m=abs(move))
                    continue
                logger.warning("[{}] seed bin NOT found — adjust the chassis "
                               "(f/b/l/r), `d` to re-detect, `q` to stop "
                               "(case still held)", label)
                if not _manual_strafe(bot, "adjust"):
                    logger.error("[{}] giving up the seed place (case still held)",
                                 label)
                    return False
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
        # corner-seat aim bias — applied HERE (not in suction.place) so the
        # reach pre-check below checks the pose that is actually FLOWN: at a
        # 50mm bias the previously-unchecked shift broke the hover approach
        # 37mm past the pre-checked column (0824)
        if label == "case":
            bias_x = bias_y = float(cfg.CASE_CORNER_AIM_BIAS_M)
        else:
            bias_x, bias_y = cfg.BATTERY_CORNER_AIM_BIAS_M
        place_pose = (place_pose[0] - np.sign(cfg.CASE_CORNER_DIR[0]) * bias_x,
                      place_pose[1] - np.sign(cfg.CASE_CORNER_DIR[1]) * bias_y,
                      *place_pose[2:])
        logger.info("[{}] corner-seat aim bias ({:+.1f},{:+.1f})mm (away from the "
                    "datum corner)", label,
                    -np.sign(cfg.CASE_CORNER_DIR[0]) * bias_x * 1000,
                    -np.sign(cfg.CASE_CORNER_DIR[1]) * bias_y * 1000)
        if not descent_reachable(mover, place_pose):
            # auto-adjust needs a detection to steer by — the no-case default
            # pose is fixed in base_link, so it goes straight to the keyboard
            if auto and tdet is not None and tdet.found and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
                adjusts += 1
                logger.warning("[{}] place pose out of reach — auto-adjust {}/{}",
                               label, adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
                dy = _auto_adjust(bot, mover, tdet, place_pose)
                if dy is not None:
                    if leg is not None:
                        leg.learn(dy, "target")
                    continue
                # nothing within the adjust clamp solves -> straight to the keyboard
            # holding the item — no auto-recovery; reposition + retry, or stop
            logger.warning("[{}] place pose out of reach — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to stop (item still held)", label)
            if not (cfg.CHASSIS_MANUAL or auto) or not _manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping on the right side (item still held)", label)
                return False
            continue
        _dual_plane_probe(bot, tgt_layers, tgt_plane, "target", label, zt)
        # misseat check only with a measured-anchored expectation (own / sibling /
        # case anchor, each with its tolerance) — the model plane has been seen off
        # by more than any of those tolerances (0804 layer 5)
        # lift_to_clear: place stops the lift at the wall-clear height (0.95, cup
        # empty) — the return strafe below starts right away and the view park
        # (target z = SAFE_TRANSPORT_Z) finishes the rise in parallel
        # The run's FIRST case releases with NO back-off: it becomes the datum
        # every later place is aligned to, so backing it 10mm off the corner
        # would move that reference. Later cases keep the back-off (preload
        # relief, so the cup retreat cannot drag the registered case).
        # The FIRST case also takes the battery's corner TIMING: straight down
        # to the first vertical contact, drive only then. Its airborne latch is
        # the least trustworthy of the run (empty bin, nothing detected to align
        # to, and it is the datum for everything after) — 0905 froze x on a
        # 3.3-5.0N spike 67-104mm above the seat and released with x unregistered.
        pres = mover.place(place_pose, expected_z=exp_z, misseat_tol_m=mtol,
                           lift_to_clear=True,
                           backoff_m=0.0 if seed else None,
                           corner_seat="case" if label == "case" else "battery",
                           corner_touch_first=seed)
        if (seed and pres is not None and pres.reason == "unreachable"
                and pres.contact_ee_z is None):
            # seed hover approach failed with the case still ON the cup and
            # nothing moved — give the operator the keyboard instead of
            # stopping the run; `d` RE-DETECTS the bin (the chassis moved, so
            # the bin-anchored pose is stale) and retries the whole place
            logger.warning("[{}] seed place unreachable (case still held) — adjust "
                           "the chassis (f/b/l/r), `d` to re-detect + retry, "
                           "`q` to stop", label)
            if _manual_strafe(bot, "adjust"):
                continue
            logger.error("[{}] stopping on the right side (case still held)", label)
            return False
        break
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

    # Fold the KNOWN offset from THIS item's ref to the NEXT item's ref into
    # the return leg itself (see run_item docstring) — the detection-based
    # centering at the next item's source only has to correct real residual.
    extra = leg.ref_change(next_y_ref) if leg is not None else 0.0
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
        strafe(bot, "left", auto, leg, extra_m=extra)
    else:
        # --- park again (clear the head view for the SOURCE detect) IN PARALLEL
        #     with the return LEFT ---
        _park_during_legs(label, [lambda: _view_park(mover, label)],
                          lambda: strafe(bot, "left", auto, leg, extra_m=extra))
    return True


def run(bot, mover: SuctionMover,
        scan: bool = False, auto: bool = False,
        src: int = cfg.SRC_LAYERS_REMAINING, tgt: int = cfg.TGT_LAYERS_REMAINING,
        final_to_bin: bool = cfg.FINAL_CASE_TO_BIN) -> bool:
    """Layer loop: each layer runs the full item set (case + 2 batteries), then
    the stacks step (source -1, target +1) so the BEV warp plane tracks the
    shrinking source / growing target. Runs src - 1 layers — the bottom case
    of the source is not a layer (no batteries), it is the last-case -> bin
    epilogue that `final_to_bin` enables. `src`/`tgt` are the STARTING
    PHYSICAL stack heights (all three come from the task menu in _main).
    `scan`/`auto` are passed through to run_item (barcode divert /
    --auto-move); with `auto`, arrival residuals feed back into learned
    per-direction leg distances (ChassisNav) used by every later strafe leg.
    The divert-case slot order (first target -> left slot, second -> right)
    is remembered HERE, across items and layers."""
    # The OPERATOR positions the chassis at the SOURCE station once (keyboard,
    # move_chassis grammar — replaces the old fixed initial left leg); each
    # item returns here at its end, so a failed pick just stops (no
    # strafe-right, robot left where it is). Residual is absorbed by centering.
    leg = ChassisNav() if auto else None
    # Divert excursion (source <-> divert case) leg: own ChassisNav instance,
    # seeded from DIVERT_CASE_STRAFE_LEFT_M — a different physical gap than
    # the main leg above, so it's calibrated independently. Its first-ever
    # visit IS a real, just-executed strafe (unlike the main leg's operator
    # start park), so it's not skip_first-protected.
    divert_leg = ChassisNav(cfg.DIVERT_CASE_STRAFE_LEFT_M, skip_first=False) if auto else None
    # measured-contact z anchors (expected z + warp plane per column), with the
    # per-run error CSV (contact-vs-predicted, plane measured-vs-model, failures)
    stamp = run_stamp()
    zt = ZTracker(log_path=None if cfg.ZTRACK_LOG_DIR is None else
                  f"{cfg.ZTRACK_LOG_DIR}/ztrack_{stamp}.csv")
    logger.info("position the chassis at the SOURCE station (case in front), then `d`")
    if not _manual_strafe(bot, "start"):
        logger.error("start positioning aborted (`q`) — run cancelled")
        return False
    # remaining divert-case slots, consumed by run_item on each successful
    # divert: first target battery -> LEFT slot, second -> RIGHT slot
    divert_slots = ["BAT_SRC_2", "BAT_SRC_1"]
    layer = 0
    try:
        # `src` is the PHYSICAL source stack. The bottom case is not a layer:
        # it has no batteries and goes to the bin box on its own (the
        # final_to_bin epilogue, picked at layers=1), so the loop moves the
        # src - 1 layers above it (src=3 -> 2 layers, then the last case).
        while src >= 2:
            layer += 1
            logger.info("=== layer {}: source stack {}, target stack {} ===", layer, src, tgt)
            for i, (label, key) in enumerate(ITEMS):
                next_key = ITEMS[(i + 1) % len(ITEMS)][1]   # cycles to CASE_PICK after battery_2
                if not run_item(bot, mover, label, key, src, tgt, next_key,
                                scan=scan, auto=auto, leg=leg, divert_leg=divert_leg, zt=zt,
                                divert_slots=divert_slots,
                                strict_start=(layer == 1 and i == 0)):
                    logger.error("stopping at layer {} item {} (robot left where it is) — "
                                 "to resume, set SRC_LAYERS_REMAINING={} TGT_LAYERS_REMAINING={}",
                                 layer, label, src, tgt)
                    return False
            src -= 1
            tgt += 1
        logger.info("=== all {} layers moved ===", layer)
        ok = True
        if final_to_bin:
            logger.info("=== final case -> bin box (strafe right {:.2f} m) ===",
                        cfg.FINAL_CASE_STRAFE_RIGHT_M)
            ok = _final_case_to_bin(bot, mover, auto, leg, zt)
        return ok
    finally:
        zt.close()
        base = cfg.CHASSIS_AUTO_STRAFE_DIST_M
        if leg is not None and leg.leg_dist_m != base:
            logger.info("learned leg distance this run: {:.3f} m "
                        "(config CHASSIS_AUTO_STRAFE_DIST_M = {:.2f})",
                        leg.leg_dist_m, base)
        divert_base = cfg.DIVERT_CASE_STRAFE_LEFT_M
        if divert_leg is not None and divert_leg.leg_dist_m != divert_base:
            logger.info("learned divert leg distance this run: {:.3f} m "
                        "(config DIVERT_CASE_STRAFE_LEFT_M = {:.2f})",
                        divert_leg.leg_dist_m, divert_base)


def run_box_discard(bot, mover: SuctionMover, gripper) -> bool:
    """--box (task 3): grip the paper box with the RIGHT gripper, carry it to
    the unload spot, set it down in front on the right.

    Both chassis legs are hand-driven (`d` when in position, `q` gives up):

      1. drive to the box -> `d`
      2. run_box_pick in "carry" mode — both arms home, BEV-detect the box,
         grip its right wall, lift to the hover; the box STAYS in the gripper
         (the same routine `python -m ik_demo.box_pick --detect --carry` runs).
         A grasp that does not solve where the box is: the smallest chassis
         move that makes it solve, re-detect, retry (CHASSIS_ADJUST_MAX_ATTEMPTS
         rounds), then the keyboard; a missed detection goes to the keyboard
      3. drive to the unload spot, box in hand -> `d`
      4. the arm lowers straight down from the hover by cfg.BOX_HOVER_HEIGHT_M
         (touchdown-guarded), OPENS, lifts back and homes. No detection and no
         gentle place: the box lands wherever the arm is, right-front.
    """
    from .go_home import safe_home
    if gripper is None:
        logger.error("--box needs the right gripper (it is what carries the box) "
                     "— none available")
        return False
    from .move_chassis import move_backward, move_forward, strafe_left, strafe_right
    logger.info("position the chassis at the BOX, then `d`")
    if not _manual_strafe(bot, "start"):
        logger.error("start positioning aborted (`q`) — --box cancelled")
        return False
    # Same ladder as the suction places: a grasp that does not solve where the
    # box is gets the SMALLEST chassis move that makes it solve
    # (GripperMover.box_reach_offset), re-detect, retry, up to
    # CHASSIS_ADJUST_MAX_ATTEMPTS; then (or on a missed detection) the keyboard.
    adjusts = 0
    while True:
        res = run_box_pick(bot, gripper, left=mover, mode="carry", home_after=False)
        if res.success:
            break
        if (res.reason == "unreachable" and res.box is not None
                and adjusts < cfg.CHASSIS_ADJUST_MAX_ATTEMPTS):
            adjusts += 1
            off = gripper.box_reach_offset(res.box)
            if off is not None:
                dx, dy = off
                logger.warning("[box] grasp out of reach — auto-adjust {}/{}: dx {:+.3f} m, "
                               "dy {:+.3f} m (nearest reachable spot)", adjusts,
                               cfg.CHASSIS_ADJUST_MAX_ATTEMPTS, dx, dy)
                if abs(dx) >= cfg.CHASSIS_ADJUST_MIN_TRANSLATE_M:
                    (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx))
                if abs(dy) >= cfg.CHASSIS_ADJUST_MIN_TRANSLATE_M:
                    (strafe_left if dy > 0 else strafe_right)(bot, distance_m=abs(dy))
                continue                    # re-detect from the new frame
            # nothing within the adjust clamp solves -> straight to the keyboard
        if res.reason in ("unreachable", "not_detected"):
            logger.warning("[box] {} — adjust the chassis (f/b/l/r), `d` to re-detect + "
                           "retry, `q` to give up", res.reason)
            if _manual_strafe(bot, "adjust"):
                continue
        logger.error("box pick failed: {} — homing the right arm, chassis stays put",
                     res.reason)
        safe_home(gripper)
        return False
    logger.info("=== drive to the UNLOAD spot by hand — the box stays in the gripper ===")
    if not _manual_strafe(bot, "unload"):
        logger.error("unload positioning aborted (`q`) — box still HELD, stopping here")
        return False
    ok = gripper.set_down_box(cfg.BOX_HOVER_HEIGHT_M)
    safe_home(gripper)
    return ok


def run_case_to_bin_here(bot, mover: SuctionMover) -> bool:
    """--case-bin (task 4): ONE case -> the bin box, with the chassis positioned
    by something else — this task drives NO chassis legs of its own. Two `d`
    triggers:

      1. chassis at the case -> `d`: BEV-detect the case (a single layer, so
         the warp plane is the one-case height), pick it, park the arm out of
         the head-camera view
      2. chassis at the bin -> `d`: BEV-detect the BIN and place the case into
         it at the empty-bin seat height (_place_case_in_bin), then home

    No bin-align strafe and no search strafes. The chassis moves on its own
    only if the place pose is out of reach: the smallest move that fixes it
    (_auto_adjust, CHASSIS_ADJUST_MAX_ATTEMPTS rounds); a missed detection
    goes straight to the operator (`adjust` prompt, `d` re-detects, `q` stops).
    """
    from .case_to_bin import _pick_source_case
    label = "case_bin"
    logger.info("chassis at the CASE, then `d`")
    if not _manual_strafe(bot, "pick"):
        logger.error("pick positioning aborted (`q`) — --case-bin cancelled")
        return False
    center = _pick_source_case(bot, mover, 1)
    if center is None:
        return False                        # pick flow already homed the arms
    _view_park(mover, label)                # clear the head camera for the bin detection
    logger.info("=== case on the cup — chassis at the BIN, then `d` ===")
    if not _manual_strafe(bot, "place"):
        logger.error("place positioning aborted (`q`) — case still HELD, stopping here")
        return False
    pres = _place_case_in_bin(bot, mover, label, center[3], 0, auto=False)
    if pres is None:
        return False
    if not getattr(pres, "success", True):
        logger.warning("[{}] bin place failed ({}) — released at the bin anyway",
                       label, pres.reason)
    _arms_home(bot, mover)
    return True


# task menu (shown when no mode flag picks one): key -> (name, flag it equals)
TASKS: tuple[tuple[str, str, str], ...] = (
    ("1", "case + battery transfer (layer by layer)", ""),
    ("2", "bin lid discard", "--lid"),
    ("3", "box discard", "--box"),
    ("4", "one case -> bin, chassis positioned externally (`d` pick, `d` place)", "--case-bin"),
)


def _choose_task() -> "str | None":
    """Interactive task menu. Returns the chosen key ("1"/"2"/"3"), None on `q`."""
    print("=" * 60)
    print("  Select the task:")
    for key, name, flag in TASKS:
        print(f"  {key}) {name}" + (f"   (= {flag})" if flag else ""))
    print("  q) quit")
    print("=" * 60)
    keys = {key for key, _, _ in TASKS}
    while True:
        s = input("task> ").strip().lower()
        if s == "q":
            return None
        if s in keys:
            return s
        print(f"enter one of {sorted(keys)} or q")


def _ask_int(prompt: str, default: int, lo: int, hi: int) -> int:
    """Integer prompt; Enter keeps the default."""
    while True:
        s = input(f"{prompt} [{default}]> ").strip()
        if not s:
            return default
        try:
            v = int(s)
        except ValueError:
            print(f"integer {lo}-{hi}, or Enter for {default}")
            continue
        if lo <= v <= hi:
            return v
        print(f"out of range {lo}-{hi}")


def _ask_yes(prompt: str, default: bool) -> bool:
    """y/n prompt; Enter keeps the default."""
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]> ").strip().lower()
    return default if not s else s in ("y", "yes")


def _ask_layers() -> tuple[int, int, bool]:
    """Task 1 parameters: starting source / target stack heights and whether
    to run the last-case -> bin epilogue. Enter on each keeps the config
    default (cfg.SRC_LAYERS_REMAINING / TGT_LAYERS_REMAINING / FINAL_CASE_TO_BIN)."""
    print("Layer setup — Enter keeps the default:")
    src = _ask_int("  source stack layers (physical height now)", cfg.SRC_LAYERS_REMAINING, 1, 8)
    tgt = _ask_int("  target stack layers at start", cfg.TGT_LAYERS_REMAINING, 1, 8)
    final = _ask_yes("  move the last case to the bin box at the end?", cfg.FINAL_CASE_TO_BIN)
    return src, tgt, final


def _announce_task(task: str, src: int, tgt: int, final_to_bin: bool) -> None:
    """Log what the chosen task is about to do with the robot. `task` is a
    TASKS key or "box-lid" (flag-only mode)."""
    logger.warning("=" * 60)
    if task == "box-lid":
        logger.warning("--box-lid: PAPER BOX LID — the normal item loop does NOT run.")
        logger.warning("BOTH chassis legs are hand-driven (`d` when in position). "
                       "1) drive to the box; the lid (box OBB class {}, warp plane "
                       "z {:.2f}) is suction-picked at its CENTRE with {:.0f}N "
                       "contact / {:.0f}N abort. 2) it is handed to the RIGHT "
                       "GRIPPER from the -y side, WHERE IT WAS PICKED (lifted "
                       "straight to z {:.2f}; the entry stops on {:.0f}N). "
                       "3) drive to the unload spot; the torso leans to {} deg "
                       "and the gripper OPENS at {}.",
                       cfg.BOX_LID_CLS_ID, cfg.BOX_LID_PLANE_Z_M,
                       cfg.BOX_LID_CONTACT_N, cfg.BOX_LID_FORCE_LIMIT_N,
                       cfg.BOX_LID_HANDOFF_Z_M, cfg.BOX_LID_SIDE_CONTACT_N,
                       cfg.LID_PLACE_TORSO_DEG, cfg.BOX_LID_DROP_EE_POS)
    elif task == "2":
        logger.warning("--lid: LID MODE — the normal item loop does NOT run.")
        logger.warning("BOTH chassis legs are hand-driven (`d` when in position). "
                       "1) drive to the lidded box; the lid (bin model class {}, warp "
                       "plane z {:.2f}) is picked {:+.0f},{:+.0f}mm from its center. "
                       "2) `d` again (back off the box first if needed): the torso goes "
                       "to {} deg and both arms stow. 3) drive to the unload spot; the lid "
                       "ON THE FLOOR is detected (plane z {:.3f}) and the held one is "
                       "stacked on it, descending to force from ee_z {:.2f}.",
                       cfg.LID_CLS_ID, cfg.LID_PLANE_Z_M,
                       cfg.LID_GRAB_OFFSET_M[0] * 1000, cfg.LID_GRAB_OFFSET_M[1] * 1000,
                       cfg.LID_PLACE_TORSO_DEG, cfg.LID_FLOOR_PLANE_Z_M,
                       cfg.LID_PLACE_START_EE_Z_M)
    elif task == "4":
        logger.warning("--case-bin: ONE CASE -> BIN — the normal item loop does NOT run.")
        logger.warning("NO chassis legs of its own (the chassis is positioned externally; "
                       "it moves only the minimum that puts an out-of-reach place inside reach). "
                       "1) `d` at the case: BEV-detect it (1 layer), suction-pick, park. "
                       "2) `d` at the bin: BEV-detect the BIN, place the case into it "
                       "(empty-bin seat ee_z {:.3f}, offset {:+.0f},{:+.0f}mm), home.",
                       cfg.FLOOR_Z_BASE_M + cfg.LAYER_PITCH_M + cfg.SUCTION_LENGTH_M,
                       cfg.BIN_PLACE_CENTER_OFFSET[0] * 1000, cfg.BIN_PLACE_CENTER_OFFSET[1] * 1000)
    elif task == "3":
        logger.warning("--box: BOX DISCARD — the normal item loop does NOT run.")
        logger.warning("BOTH chassis legs are hand-driven (`d` when in position). "
                       "1) drive to the box; the RIGHT arm grips its right wall "
                       "(BEV detect, rim z {:.2f}) and lifts it {:.0f}cm to the hover. "
                       "2) drive to the unload spot with the box in the gripper. "
                       "3) the arm lowers {:.0f}cm straight down, OPENS, homes — the "
                       "box lands right-front, wherever the arm is.",
                       cfg.BOX_RIM_Z_M, cfg.BOX_HOVER_HEIGHT_M * 100, cfg.BOX_HOVER_HEIGHT_M * 100)
    else:
        logger.warning("MOVES THE REAL ARM + SUCTION + CHASSIS (strafe L/R per item):")
        for label, key in ITEMS:
            logger.warning("   {} ({})", label, key)
        logger.warning("Layers: source {} -> target {} at start = {} case+battery layer(s), "
                       "then the last case -> bin box: {}",
                       src, tgt, max(src - 1, 0), "YES" if final_to_bin else "no")
        logger.warning("Barcode divert (target codes -> divert case, {:.2f} m LEFT of the "
                       "source; first -> left slot, second -> right).",
                       cfg.DIVERT_CASE_STRAFE_LEFT_M)
        logger.warning("AUTO chassis: {:.2f} m legs + detection-based adjust.",
                       cfg.CHASSIS_AUTO_STRAFE_DIST_M)
    logger.warning("=" * 60)


def _main() -> None:
    from dexcontrol.core.config import get_robot_config

    KNOWN_FLAGS = {"--dashboard", "--state-publish",
                   "--box", "--lid", "--box-lid", "--case-bin"}
    unknown = [a for a in sys.argv[1:] if a not in KNOWN_FLAGS]
    if unknown:
        raise ValueError(f"unknown flag(s) {unknown} — choose from {sorted(KNOWN_FLAGS)}")

    # Barcode divert and automatic chassis legs are ALWAYS on (they used to be
    # --gripper / --auto-move); run()'s `scan` / `auto` still exist for callers.
    use_gripper = True
    auto_move = True
    use_dashboard = "--dashboard" in sys.argv  # spool camera/joints/EE/wrench (as sequence.py)
    use_state_publish = "--state-publish" in sys.argv  # mirror joint state to zenoh (e.g. for Isaac mirror)
    # A mode flag picks the FIRST task without the menu; every later task comes
    # from the menu (the session loops back to it after each task, `q` ends it).
    first_task = ("2" if "--lid" in sys.argv else            # bin lid discard
                  "3" if "--box" in sys.argv else            # box discard (right gripper)
                  "4" if "--case-bin" in sys.argv else       # one case -> bin, no own strafes
                  "box-lid" if "--box-lid" in sys.argv else  # paper box lid handoff (flag only)
                  None)

    setup_logging()
    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARMS + SUCTION + GRIPPER + CHASSIS. One robot session, "
                   "many tasks: the task menu comes back after each task (`q` ends).")
    if use_dashboard:
        logger.warning("Dashboard spool ENABLED — view with run_dashboard_demo.sh.")
    if use_state_publish:
        logger.warning("Zenoh joint-state publish ENABLED (arm/head/torso -> state_publish).")
    logger.warning("Corner-seat places (experimental, unverified): case driven into the "
                   "bin corner while descending (aim {:+.0f},{:+.0f}mm off); battery "
                   "driven into the slot corner after contact (aim {:+.0f},{:+.0f}mm off).",
                   -np.sign(cfg.CASE_CORNER_DIR[0]) * cfg.CASE_CORNER_AIM_BIAS_M * 1000,
                   -np.sign(cfg.CASE_CORNER_DIR[1]) * cfg.CASE_CORNER_AIM_BIAS_M * 1000,
                   -np.sign(cfg.CASE_CORNER_DIR[0]) * cfg.BATTERY_CORNER_AIM_BIAS_M[0] * 1000,
                   -np.sign(cfg.CASE_CORNER_DIR[1]) * cfg.BATTERY_CORNER_AIM_BIAS_M[1] * 1000)
    logger.warning("Clear the strafe path. E-stop in reach.")
    logger.warning("=" * 60)
    if input("Connect + safe-home both arms? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        # Arm + chassis state subscribers ALWAYS_ON. Their default AUTO policy
        # undeclares the zenoh liveliness token after 5 s without a read
        # (dexcontrol IdleMonitor), which every interactive prompt in this run
        # sits through — the `y/N` gates here, and `d`/`q` on each CHASSIS_MANUAL
        # leg. resume() then CLEARS the subscriber buffer, so the first read
        # after a prompt (joint pos for IK, wrench for descend-to-contact) waits
        # on fresh data or comes back None. recursive covers wrench / wrist
        # button / motor temperatures and the chassis steer + drive pair.
        for part in (bot.left_arm, bot.right_arm, bot.chassis, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        logger.info("arm + chassis subscribers pinned always_on (no idle pause)")
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        publisher = None
        if use_dashboard:
            from .dashboard_publish import DashboardPublisher
            publisher = DashboardPublisher(bot).start()
        state_publisher = None
        if use_state_publish:
            from .state_publish import StatePublisher
            state_publisher = StatePublisher(bot).start()
        try:
            with SuctionMover(bot) as m:
                release = m.software_estop_active()
                if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                    return
                if not m.ensure_ready(release_estop=release):
                    logger.error("arm not ready — aborting")
                    return
                from .go_home import both_arms_home, safe_home
                # --- PRELOAD, once per session. These used to happen inside the
                #     tasks: on task 3 the right-arm model build (twice: construct
                #     + the ensure_ready re-pin) and the Robotiq init cost ~10 s
                #     after `Start?`, and every task's first detection paid a
                #     YOLO load (0907 13:50 log). The detectors load in a thread
                #     while the right arm builds and both arms home here.
                t0 = time.monotonic()
                detectors = threading.Thread(target=_preload_detectors,
                                             name="preload-detectors", daemon=True)
                detectors.start()
                gripper = GripperMover(bot)   # the right arm's mover, with or without a Robotiq
                _RIGHT_ARM[:] = [gripper]     # lid stance / restore reuse it
                if not (gripper.ensure_ready() and gripper.initialize()):
                    # the arm itself still homes and parks; only the two modes
                    # that carry something in the Robotiq are off
                    logger.warning("right gripper not ready — tasks 3 (--box) and --box-lid "
                                   "cannot run this session")
                # Start clean: BOTH arms safe-homed (lift-if-low first, so an arm
                # left down in a box by a previous run doesn't sweep the walls).
                logger.info("-> both arms safe home")
                both_arms_home(bot, left=m, right=gripper)
                detectors.join()
                logger.info("preload done in {:.1f}s (right arm + gripper + detectors)",
                            time.monotonic() - t0)
                task = first_task
                while True:
                    if task is None:
                        task = _choose_task()
                        if task is None:
                            break
                    src, tgt, final_to_bin = (cfg.SRC_LAYERS_REMAINING, cfg.TGT_LAYERS_REMAINING,
                                              cfg.FINAL_CASE_TO_BIN)
                    if task == "1":
                        src, tgt, final_to_bin = _ask_layers()
                    _announce_task(task, src, tgt, final_to_bin)   # no gate: the menu choice is the go
                    # Tilt the head down so the box is in view (same as live_bev/
                    # capture); the BEV homography uses the live joints, so ~24 deg
                    # matches the training data. Per task: the lid modes leave it
                    # at LID_PLACE_HEAD_PITCH_DEG.
                    set_head_pitch(bot, angle=24.0)
                    if task in ("3", "box-lid"):
                        # The right-arm model was built at session start from
                        # the torso stance then; a lid task that died leaned
                        # over leaves the torso elsewhere. Re-pin (torso back to
                        # the taught stance + model rebuild) only in that case.
                        live = np.asarray(bot.torso.get_joint_pos(), dtype=float)
                        if np.max(np.abs(live - gripper._torso_q)) > 0.02:
                            logger.info("torso moved since the right-arm model was built "
                                        "— re-pinning before the gripper task")
                            gripper.ensure_ready()
                    # tasks that carry something in the Robotiq need it answering
                    robotiq = gripper if gripper.gripper is not None else None
                    if task == "box-lid":
                        ok = run_box_lid(bot, m, robotiq)
                    elif task == "2":
                        ok = run_lid(bot, m)
                    elif task == "3":
                        ok = run_box_discard(bot, m, robotiq)
                    elif task == "4":
                        ok = run_case_to_bin_here(bot, m)
                    else:
                        ok = run(bot, m, scan=use_gripper, auto=auto_move,
                                 src=src, tgt=tgt, final_to_bin=final_to_bin)
                    # safe_home, not a bare move_joints: this runs after a FAILURE
                    # too, where the arm is wherever the failed leg halted it, and a
                    # joint move home from down in a box sweeps the walls — the very
                    # thing the start-of-run both_arms_home above guards against. It
                    # was the one home path in the file still unguarded (0905 15:30:
                    # the lid pick died at ee_z 1.0923 and the arm went straight home
                    # from a near-full-extension pose, swinging wide to the left).
                    # NOTE this only covers the LOW case: HOME_LIFT_MIN_EE_Z is 1.10,
                    # so that 1.0923 buys a 9mm lift and the same joint swing follows.
                    # An EXTENDED-pose guard (retract in x before homing) is the open
                    # half of the problem.
                    #
                    # The lid modes END at the unload station, torso leaned to
                    # LID_PLACE_TORSO_DEG with the arm out over open floor: the
                    # z=1.10 target is 200mm+ out of reach there (0906 10:25 logged
                    # a 206.3mm shortfall) and there is no box wall to clear, so the
                    # lift is skipped. It stays ON for the case modes, whose failures
                    # do strand the cup down in a box.
                    safe_home(m, lift=task not in ("2", "box-lid"))
                    logger.info("task {} {} — back to the menu", task, "OK" if ok else "FAILED")
                    task = None
        finally:
            if publisher is not None:
                publisher.stop()
            if state_publisher is not None:
                state_publisher.stop()


if __name__ == "__main__":
    _main()
