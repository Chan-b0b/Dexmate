"""Detection-driven case_pick episode collector for VLA training data.

One keyboard-gated cycle per episode:

    ENTER -> BEV-detect the case -> record ON -> pick (approach@transport ->
    hover -> descend-to-contact -> seal -> lift) -> record OFF -> suction OFF
    -> wait 2 seconds -> return home -> wait for the next ENTER.

Move the box / restack layers between cycles for position + depth diversity;
the fresh detection each cycle re-centers everything (that variation is the
point — it decorrelates contact depth from the habitual ee_z).

Every cycle is a press-retreat pick (SuctionMover.pick_retreat): suction ON
from creep_z as usual, press on to a per-episode random force (TOUCH_N_RANGE,
tared vertical), retreat RETREAT_M_RANGE (per-episode random) above the
touched surface, and HOVER there
(no re-descent) until the vacuum grabs the case and seals — then lift
immediately. Every take thus shows force-rise -> stop -> +dz before the seal
— the retreat reflex press-hold-seal-only demos never show (policies trained
without it keep descending on contact and over-press; see
experiment_docs/EXPERIMENTS_CASE_PICK_0708.md §6.8). Sampled params land in
meta.json under "retreat".

Takes are written in the exact case_battery_demo recorder format, so the
existing converter ingests them unchanged:

    LGES/recordings/<YYYYMMDD>/case_pick/<YYYYmmdd-HHMMSS_epNNNN_case_pick>/
        meta.json                # instruction, success, detection IVs
        head_rgb/000000.jpg ...
        head_depth/000000.png    # 16-bit millimetres (0 = invalid)
        states.jsonl             # i, t, joints, ee(pos+quat), wrench,
                                 # suction_cmd, vacuum_sealed

    /home/dexmate/vla_venv/bin/python LGES/vla_training/convert_to_lerobot.py \
        --recordings LGES/recordings/<YYYYMMDD> --tasks case_pick --name <dataset>

Review collected takes in a browser (gallery + frame scrubber, port 8081):

    cd LGES && python -m case_battery_demo.dashboard.review_server \
        --root vla_training/recordings

NOTE: suction_cmd is read from ik_demo's OWN drivers.suction_io module flag
(the case_battery_demo recorder reads its own package's flag, which this demo
never sets — reusing it would log suction_cmd=False on every frame).

Run from the repo root:

    python -m LGES.vla_training.collect_case_pick
    # or:  python LGES/vla_training/collect_case_pick.py
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sys

import cv2
import numpy as np
import pinocchio as pin
from loguru import logger
from scipy.spatial.transform import Rotation

# Allow `python LGES/vla_training/collect_case_pick.py` (repo root on path so
# LGES.ik_demo resolves); harmless under `python -m LGES.vla_training....`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LGES.ik_demo import config as cfg  # noqa: E402
from LGES.ik_demo.config import resolve_poses  # noqa: E402
from LGES.ik_demo.suction import SuctionMover  # noqa: E402
from LGES.ik_demo.drivers import suction_io  # noqa: E402
from LGES.ik_demo.chassis_sequence import (detect, _center_from_det,  # noqa: E402
                                           descent_reachable, _view_park,
                                           _center_case, _auto_adjust)
from LGES.ik_demo.move_chassis import move_forward, move_backward  # noqa: E402

RECORD_HZ = 15.0
TOUCH_N_RANGE = (12.0, 15.0)      # press force before the retreat (tared vertical N)
RETREAT_M_RANGE = (0.005, 0.008)  # contact-referenced suction-hover height (m)
CENTER_TOL_M = 0.04   # tighter than ik_demo's shared CHASSIS_CENTER_TOL_M (0.05 m)
X_CENTER_TOL_M = 0.04      # forward/back deadband vs SOURCE_CASE_CENTER x
X_CENTER_MAX_MOVES = 5     # matches ik_demo's CHASSIS_CENTER_MAX_MOVES
_KST = timezone(timedelta(hours=9))
OUT_DIR = (Path(__file__).resolve().parents[1] / "recordings"
           / datetime.now(_KST).strftime("%Y%m%d"))
PHASE = "case_pick_val"
INSTRUCTION = "pick up the case with the suction cup"  # must match training phrasing


class _EEKin:
    """FK for the suction arm's EE frame in base_link (full URDF, live torso +
    arm joints) — same convention the original training data was recorded with.
    Owns its own pinocchio data, independent of the mover's IK (thread-safe)."""

    def __init__(self) -> None:
        rw = pin.RobotWrapper.BuildFromURDF(
            filename=cfg.URDF_PATH,
            package_dirs=[
                os.path.dirname(cfg.URDF_PATH),
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cfg.URDF_PATH)))),
            ],
            root_joint=None,
        )
        self._model = rw.model
        self._data = self._model.createData()
        self._fid = self._model.getFrameId(cfg.EE_FRAME)
        prefix = "L" if cfg.ARM_SIDE == "left" else "R"
        self._arm_jids = [self._model.getJointId(f"{prefix}_arm_j{i + 1}") for i in range(7)]
        self._torso_jids = [self._model.getJointId(f"torso_j{i + 1}") for i in range(3)]

    def compute(self, torso_q, arm_q) -> tuple[np.ndarray, np.ndarray]:
        """(position [x,y,z] m, rpy [r,p,y] rad) of the EE frame in base_link."""
        q = pin.neutral(self._model)
        for jid, v in zip(self._torso_jids, np.asarray(torso_q, dtype=float)):
            q[self._model.idx_qs[jid]] = v
        for jid, v in zip(self._arm_jids, np.asarray(arm_q, dtype=float)):
            q[self._model.idx_qs[jid]] = v
        pin.framesForwardKinematics(self._model, self._data, q)
        T = self._data.oMf[self._fid]
        return T.translation.copy(), Rotation.from_matrix(T.rotation).as_euler("xyz")


def _rpy_to_quat_wxyz(rpy) -> list[float]:
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return [float(w), float(x), float(y), float(z)]


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class EpisodeRecorder:
    """15 Hz sampler thread writing takes in the case_battery_demo format.

    begin()/end() are called from the main thread between motions; the sampler
    thread reads camera + joints + wrench + suction each tick and writes frames
    only while a take is open. Takes stream into <out>/.pending and are moved
    into <out>/case_pick/ on end() (crash leaves only .pending debris).
    """

    def __init__(self, bot, out_dir: Path, hz: float = RECORD_HZ) -> None:
        self._bot = bot
        self._out = Path(out_dir)
        self._pending = self._out / ".pending"
        self._period = 1.0 / hz
        self._fk = _EEKin()
        self._vac: suction_io.VacuumMonitor | None = None
        self._lock = threading.Lock()
        self._cur: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Resume episode numbering AND the takes counter from the takes already
        # in this date folder (a second run the same day must not restart at
        # ep0001 / takes=0).
        done = self._out / PHASE
        existing = [int(m.group(1)) for p in (done.iterdir() if done.is_dir() else ())
                    if (m := re.search(r"_ep(\d+)_", p.name))]
        self._ep_index = max(existing, default=0)
        self.kept = len(existing)
        self._last_final: Path | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> "EpisodeRecorder":
        self._pending.mkdir(parents=True, exist_ok=True)
        try:
            self._vac = suction_io.VacuumMonitor()
            self._vac.start()
        except Exception as e:  # noqa: BLE001
            logger.warning("[record] vacuum monitor unavailable: {} (seal -> null)", e)
            self._vac = None
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        logger.info("[record] ready — takes -> {}", self._out / PHASE)
        return self

    def stop(self) -> None:
        if self._cur is not None:
            self.end(success=False, extra={"aborted": True})
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._vac is not None:
            try:
                self._vac.stop()
            except Exception:  # noqa: BLE001
                pass

    # -- episode control (main thread) ----------------------------------------

    def begin(self, extra: dict | None = None, tag: str | None = None) -> str:
        """Open a take; frames stream until end(). `extra` lands in meta.json
        (detection center / stack height — the IVs the research analysis needs).
        `tag` is appended to the take name (e.g. "L3" = stack layers)."""
        self._ep_index += 1
        name = f"{datetime.now(_KST).strftime('%Y%m%d-%H%M%S')}_ep{self._ep_index:04d}_{PHASE}"
        if tag:
            name += f"_{tag}"
        path = self._pending / name
        (path / "head_rgb").mkdir(parents=True)
        (path / "head_depth").mkdir(parents=True)
        cur = {
            "name": name, "path": path, "idx": 0, "t0": time.time(),
            "states": open(path / "states.jsonl", "w"),
            "extra": dict(extra or {}),
        }
        with self._lock:
            self._cur = cur
        logger.info("[record] ● recording {}", name)
        return name

    def end(self, success: bool, extra: dict | None = None) -> None:
        with self._lock:
            cur, self._cur = self._cur, None
        if cur is None:
            return
        cur["states"].close()
        cur["extra"].update(extra or {})
        meta = {
            "name": cur["name"],
            "created": datetime.now(_KST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "instruction": INSTRUCTION,
            "phase": PHASE,
            "success": bool(success),
            "arm_side": cfg.ARM_SIDE,
            "ee_frame": cfg.EE_FRAME,
            "urdf": cfg.URDF_PATH,
            "action_space": "ee_delta+suction",
            "episode_scope": "one detection-driven pick (ik_demo collect_case_pick)",
            "ee_rotation": "quat_wxyz, base_link",
            "depth_units": "uint16 millimetres (0 = invalid)",
            "frames": cur["idx"],
            "duration_s": round(time.time() - cur["t0"], 3),
        }
        meta.update(cur["extra"])
        _atomic_write(str(cur["path"] / "meta.json"), json.dumps(meta, indent=2).encode())
        final = self._out / PHASE / cur["name"]
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(cur["path"], final)
        self.kept += 1
        self._last_final = final
        logger.success("[record] ✓ kept {} ({} frames, success={})", cur["name"], cur["idx"], success)

    def discard_last(self) -> None:
        if self._last_final is None or not self._last_final.is_dir():
            logger.warning("[record] no saved take to discard")
            return
        shutil.rmtree(self._last_final, ignore_errors=True)
        logger.warning("[record] ✗ discarded {}", self._last_final.name)
        self.kept = max(0, self.kept - 1)
        self._ep_index = max(0, self._ep_index - 1)
        self._last_final = None

    # -- sampler thread --------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            with self._lock:
                cur = self._cur
                if cur is not None:
                    try:
                        self._write_frame(cur)
                    except Exception as e:  # noqa: BLE001 — one bad read must not kill the take
                        logger.debug("[record] frame error: {}", e)
            self._stop.wait(max(0.0, self._period - (time.time() - t0)))

    def _write_frame(self, cur: dict) -> None:
        stamp = time.time()
        cam = self._bot.sensors.head_camera
        rgb, depth = cam.get_left_rgb(), cam.get_depth()
        i = cur["idx"]
        stem = f"{i:06d}"
        if rgb is not None:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                _atomic_write(str(cur["path"] / "head_rgb" / (stem + ".jpg")), buf.tobytes())
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            mm = np.clip(np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0, neginf=0.0), 0, 65535)
            cv2.imwrite(str(cur["path"] / "head_depth" / (stem + ".png")), mm.astype(np.uint16))

        joints: dict[str, dict[str, float]] = {}
        for comp in ("left_arm", "right_arm", "torso", "head"):
            try:
                if self._bot.has_component(comp):
                    jp = getattr(self._bot, comp).get_joint_pos_dict()
                    joints[comp] = {k: float(v) for k, v in jp.items()}
            except Exception:  # noqa: BLE001
                pass

        arm = self._bot.left_arm if cfg.ARM_SIDE == "left" else self._bot.right_arm
        pos, rpy = self._fk.compute(self._bot.torso.get_joint_pos(), arm.get_joint_pos())

        wrench = None
        ws = getattr(arm, "wrench_sensor", None)
        if ws is not None:
            try:
                raw = np.asarray(ws.get_state()["wrench"], dtype=float)
                fx, fy, fz, tx, ty, tz = (float(v) for v in raw[:6])
                wrench = {"fx": fx, "fy": fy, "fz": fz, "tx": tx, "ty": ty, "tz": tz,
                          "raw_mag": float(np.linalg.norm(raw[:3]))}
            except Exception:  # noqa: BLE001
                pass

        sealed = None
        if self._vac is not None:
            try:
                sealed = bool(self._vac.is_sealed())
            except Exception:  # noqa: BLE001
                pass

        row = {
            "i": i,
            "t": stamp,
            "joints": joints,
            "wrench": wrench,
            "suction_cmd": suction_io.is_suction_commanded_on(),
            "vacuum_sealed": sealed,
            "barcode_confirmed": False,
            "gripper_pos": None,
            "ee": {"frame": cfg.EE_FRAME, "pos": [float(v) for v in pos],
                   "quat_wxyz": _rpy_to_quat_wxyz(rpy)},
        }
        cur["states"].write(json.dumps(row) + "\n")
        cur["idx"] = i + 1


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def _recover_to_transport(mover: SuctionMover) -> None:
    """After a failed pick (cup halted low, suction already off) lift straight
    up to transport height before parking."""
    pos, rpy = mover.current_ee_pose()
    if float(pos[2]) < cfg.SAFE_TRANSPORT_Z - 0.01:
        mover.move_ee([float(pos[0]), float(pos[1]), cfg.SAFE_TRANSPORT_Z], tuple(rpy), quiet=True)


def _cycle(bot, mover: SuctionMover, rec: EpisodeRecorder, layers: int) -> None:
    # Tight chassis centering on the detected case (yaw -> 0, y -> CENTER_TOL_M
    # of CHASSIS_CENTER_CASE_Y_M) before computing the pick pose, same as
    # ik_demo's run_item — then a reach-check + auto-adjust fallback in case
    # centering alone still leaves the pose out of reach.
    y_ref = cfg.CHASSIS_CENTER_CASE_Y_M - resolve_poses((0.0, 0.0, 0.0, 0.0))["CASE_PICK"][1]
    det = _center_case(bot, layers, "collect", "source", None, y_ref, tol_m=CENTER_TOL_M)
    if det is None or not det.found:
        logger.warning("no case detected — reposition and try again")
        return
    # _center_case only handles yaw + lateral (y); close the forward/back (x)
    # gap too, same dx convention as chassis_sequence._auto_adjust (+ = case
    # too far ahead -> move forward).
    for _ in range(X_CENTER_MAX_MOVES):
        dx = det.base_xy[0] - cfg.SOURCE_CASE_CENTER[0]
        if abs(dx) <= X_CENTER_TOL_M:
            break
        dx = float(np.clip(dx, -cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M, cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M))
        logger.info("centering case x: {} {:.3f} m (ref x {:.2f})",
                    "forward" if dx > 0 else "backward", abs(dx), cfg.SOURCE_CASE_CENTER[0])
        (move_forward if dx > 0 else move_backward)(bot, distance_m=abs(dx))
        det = detect(bot, layers)
        if det is None or not det.found:
            logger.warning("case lost during x-centering — reposition and try again")
            return
    adjusts = 0
    while True:
        center = _center_from_det(det)
        logger.info("case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg conf={:.2f} top_face_z={:.4f}",
                    det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.conf, det.top_face_z)
        pick_pose = resolve_poses(center)["CASE_PICK"]
        if descent_reachable(mover, pick_pose):
            break
        if adjusts >= cfg.CHASSIS_ADJUST_MAX_ATTEMPTS:
            logger.warning("pick pose out of reach after {} auto-adjusts — move the "
                           "case/chassis and try again", adjusts)
            return
        adjusts += 1
        logger.warning("pick pose out of reach — auto-adjust {}/{}",
                       adjusts, cfg.CHASSIS_ADJUST_MAX_ATTEMPTS)
        _auto_adjust(bot, det, (cfg.SOURCE_CASE_CENTER[0], y_ref))
        det = detect(bot, layers)
        if det is None or not det.found:
            logger.warning("case lost after auto-adjust — reposition and try again")
            return
    # Layer-INDEPENDENT descent: feed pick() the max-stack contact z regardless
    # of the detected layer, so the fast->creep handoff — where suction turns
    # ON — happens at the SAME EE z (SOURCE_CASE_CENTER z + DESCENT_CREEP_GAP)
    # on every episode. The creep then runs until real contact/seal, so the phase
    # transition in the data is driven by contact, not a vision-correlated
    # height. (Assumes stacks of <= 5 layers; the detected z is still recorded
    # below for analysis. Worst-case creep, layer 1: ~10 cm, ~5 s.)
    exp_z = float(cfg.SOURCE_CASE_CENTER[2])

    # Per-episode random press force and hover gap so neither is a single
    # memorizable value (6-15N spans the FiLM contact channel's F0=6..
    # saturation range, below FORCE_HARD_LIMIT_N=20).
    touch_n = round(random.uniform(*TOUCH_N_RANGE), 2)
    retreat_m = round(random.uniform(*RETREAT_M_RANGE), 4)
    rec.begin(tag=f"L{layers}", extra={
        "detected_center_xyzyaw": [float(v) for v in center],
        "top_face_z": float(det.top_face_z),
        "detect_conf": float(det.conf),
        "layers_remaining": int(layers),
        "expected_ee_z": float(exp_z),
        "retreat": {"touch_n": touch_n, "retreat_m": retreat_m},
    })
    res = mover.pick_retreat(pick_pose, expected_z=exp_z,
                             touch_n=touch_n, retreat_m=retreat_m)
    rec.end(success=bool(res.success),
            extra={"pick_reason": res.reason,
                   "contact_ee_z": None if res.contact_ee_z is None else float(res.contact_ee_z)})

    if res.success:
        logger.info("pick complete — suction off, then home")
        suction_io.suction_off()
        time.sleep(cfg.CASE_PICK_RELEASE_WAIT_S)
        mover.move_joints(mover._home_seed)
    else:
        logger.warning("pick failed: {} — recovering to transport height", res.reason)
        _recover_to_transport(mover)
        _view_park(mover, "collect")


def _main() -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot
    from LGES.ik_demo.go_home import both_arms_home

    # perception/utils was put on sys.path by the chassis_sequence import above.
    from utils import set_head_pitch

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL ARM + SUCTION. Each ENTER runs one recorded")
    logger.warning("case pick, then releases it and returns home. Move the box between")
    logger.warning("cycles for position/depth diversity. E-stop in reach.")
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
        set_head_pitch(bot, angle=15.0)  # BEV homography expects ~15 deg
        with SuctionMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                logger.error("arm not ready — aborting")
                return
            logger.info("-> both arms safe home")
            both_arms_home(bot, left=m)
            _view_park(m, "collect")

            rec = EpisodeRecorder(bot, OUT_DIR).start()
            layers = int(cfg.SRC_LAYERS_REMAINING)
            try:
                while True:
                    cmd = input(f"[collect] takes={rec.kept} layers={layers} — "
                                "ENTER=cycle · l N=set layers · d=discard last · q=quit > ").strip().lower()
                    if cmd == "q":
                        break
                    if cmd == "d":
                        rec.discard_last()
                        continue
                    if cmd.startswith("l"):
                        try:
                            layers = int(cmd.split()[1])
                        except (IndexError, ValueError):
                            logger.warning("usage: l N   (stack layers for the BEV warp plane)")
                        continue
                    if cmd == "":
                        _cycle(bot, m, rec, layers)
            finally:
                rec.stop()
            logger.info("-> home ({} takes kept)", rec.kept)
            m.move_joints(m._home_seed)


if __name__ == "__main__":
    _main()
