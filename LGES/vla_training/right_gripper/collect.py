"""Keyboard-taught right-gripper pick episode collector (force-pick VLA data).

One cycle per episode:

    ENTER -> (object location unknown: arm to DEFAULT_GRIP_POSE, YOU jog to
    the object, ENTER freezes the grasp pose, retreat + home) -> record ON ->
    IK approach (standoff -> in) -> close at the session grip force -> lift ->
    hold -> record OFF -> put down at a RANDOM spot inside the PLACE box ->
    label (y/n/d) -> home.

The random put-down spot becomes the NEXT episode's grasp target, so after
the first taught pick the loop runs jog-free with free position diversity
('j' at the prompt forces a re-jog; a failed grasp also falls back to jog).
Vary the grip force between episodes ('f N') — the force IV plus the success
label ("did the egg survive") is what the force-modulation training needs.

Jog keys (base frame, same bindings as case_battery_demo.teach_pose):
    w / s : +x / -x      a / d : +y / -y      r / f : +z / -z
    u / o : roll +/-     i / k : pitch +/-    j / l : yaw +/-
    + / - : bigger / smaller position step
    ENTER : grasp here          q : abort cycle

Takes are written in the case_battery_demo recorder format (head_rgb/,
head_depth/, states.jsonl, meta.json) so the existing tooling reads them;
each frame additionally carries the live Robotiq status under "gripper"
(gPO position, gCU motor current ~10 mA/count, gOBJ object flag) and the
commanded pos/speed/force under "gripper_cmd":

    LGES/recordings/<YYYYMMDD>/gripper_pick/<...ep NNNN_gripper_pick>/

Run from the repo root (serial port needs dialout membership):

    python -m LGES.vla_training.right_gripper.collect
    # or:  python LGES/vla_training/right_gripper/collect.py
"""

# sudo chmod 666 /dev/ttyUSB0

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from loguru import logger
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
# Collision/ FIRST: it shadows the repo-root collision_monitor.py prototype
# with the model-based (gravity+friction residual) monitor.
sys.path.insert(0, str(_REPO_ROOT / "Collision"))

from LGES.ik_demo import config as ikcfg  # noqa: E402
from LGES.ik_demo.arm import ArmMover  # noqa: E402
from LGES.ik_demo.drivers.robotiq_usb import RobotiqGripperUSB  # noqa: E402
from LGES.vla_training.right_gripper import config as rcfg  # noqa: E402
from collision_monitor import CollisionMonitor  # noqa: E402  (Collision/, model-based)

_KST = timezone(timedelta(hours=9))
OUT_DIR = (Path(__file__).resolve().parents[2] / "recordings"
           / datetime.now(_KST).strftime("%Y%m%d"))


# ---------------------------------------------------------------------------
# FK for the right-gripper EE frame (same pattern as collect_case_pick._EEKin)
# ---------------------------------------------------------------------------
class _EEKin:
    """FK for R_gripper_base in base_link (full URDF, live torso + arm joints).
    Owns its own pinocchio data, independent of the mover's IK (thread-safe)."""

    def __init__(self) -> None:
        rw = pin.RobotWrapper.BuildFromURDF(
            filename=ikcfg.URDF_PATH,
            package_dirs=[
                os.path.dirname(ikcfg.URDF_PATH),
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ikcfg.URDF_PATH)))),
            ],
            root_joint=None,
        )
        self._model = rw.model
        self._data = self._model.createData()
        self._fid = self._model.getFrameId(ikcfg.GRIPPER_EE_FRAME)
        self._arm_jids = [self._model.getJointId(f"R_arm_j{i + 1}") for i in range(7)]
        self._torso_jids = [self._model.getJointId(f"torso_j{i + 1}") for i in range(3)]

    def compute(self, torso_q, arm_q) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Episode recorder (case_battery_demo take format + gripper channel)
# ---------------------------------------------------------------------------
class EpisodeRecorder:
    """RECORD_HZ sampler thread; frames stream only while a take is open.

    Concurrent Modbus status reads are safe: RobotiqGripperUSB serialises
    transactions behind its own lock, so this thread's read_status() just
    interleaves with the main thread's goto()/wait polling.
    """

    def __init__(self, bot, gripper: RobotiqGripperUSB, out_dir: Path,
                 hz: float = rcfg.RECORD_HZ) -> None:
        self._bot = bot
        self._gripper = gripper
        self._out = Path(out_dir)
        self._pending = self._out / ".pending"
        self._period = 1.0 / hz
        self._fk = _EEKin()
        self._lock = threading.Lock()
        self._cur: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Last commanded gripper move (main thread writes, sampler reads).
        self.gripper_cmd: dict = {"pos": None, "speed": None, "force": None}
        done = self._out / rcfg.PHASE
        self._ep_index = max(
            (int(m.group(1)) for p in (done.iterdir() if done.is_dir() else ())
             if (m := re.search(r"_ep(\d+)_", p.name))),
            default=0,
        )
        self.kept = 0
        self._last_final: Path | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> "EpisodeRecorder":
        self._pending.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        logger.info("[record] ready — takes -> {}", self._out / rcfg.PHASE)
        return self

    def stop(self) -> None:
        if self._cur is not None:
            self.end(success=False, extra={"aborted": True})
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- episode control (main thread) ----------------------------------------

    def begin(self, instruction: str, extra: dict | None = None) -> str:
        self._ep_index += 1
        name = f"{datetime.now(_KST).strftime('%Y%m%d-%H%M%S')}_ep{self._ep_index:04d}_{rcfg.PHASE}"
        path = self._pending / name
        (path / "head_rgb").mkdir(parents=True)
        (path / "head_depth").mkdir(parents=True)
        cur = {
            "name": name, "path": path, "idx": 0, "t0": time.time(),
            "states": open(path / "states.jsonl", "w"),
            "instruction": instruction,
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
            "instruction": cur["instruction"],
            "phase": rcfg.PHASE,
            "success": bool(success),
            "arm_side": "right",
            "ee_frame": ikcfg.GRIPPER_EE_FRAME,
            "urdf": ikcfg.URDF_PATH,
            "action_space": "ee_delta+gripper",
            "episode_scope": "one keyboard-taught gripper pick (right_gripper collect)",
            "ee_rotation": "quat_wxyz, base_link",
            "depth_units": "uint16 millimetres (0 = invalid)",
            "frames": cur["idx"],
            "duration_s": round(time.time() - cur["t0"], 3),
        }
        meta.update(cur["extra"])
        _atomic_write(str(cur["path"] / "meta.json"), json.dumps(meta, indent=2).encode())
        final = self._out / rcfg.PHASE / cur["name"]
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(cur["path"], final)
        self.kept += 1
        self._last_final = final
        logger.success("[record] ✓ kept {} ({} frames, success={})", cur["name"], cur["idx"], success)

    def relabel_last(self, success: bool, extra: dict | None = None) -> None:
        """Rewrite the last take's meta with the human label (crush/success is
        judged after the motion finished, once the take is already closed)."""
        if self._last_final is None or not (self._last_final / "meta.json").is_file():
            logger.warning("[record] no saved take to relabel")
            return
        meta_path = self._last_final / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["success"] = bool(success)
        meta.update(extra or {})
        _atomic_write(str(meta_path), json.dumps(meta, indent=2).encode())
        logger.info("[record] relabelled {} success={}", self._last_final.name, success)

    def amend_last(self, extra: dict) -> None:
        """Merge post-episode facts (e.g. the random put-down position — it
        only exists after the take has closed) into the last take's meta."""
        if self._last_final is None or not (self._last_final / "meta.json").is_file():
            return
        meta_path = self._last_final / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta.update(extra)
        _atomic_write(str(meta_path), json.dumps(meta, indent=2).encode())

    def discard_last(self) -> None:
        if self._last_final is None or not self._last_final.is_dir():
            logger.warning("[record] no saved take to discard")
            return
        shutil.rmtree(self._last_final, ignore_errors=True)
        logger.warning("[record] ✗ discarded {}", self._last_final.name)
        self.kept = max(0, self.kept - 1)
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

        arm = self._bot.right_arm
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

        grip = None
        try:
            s = self._gripper.read_status()
            if s is not None:
                grip = {"gPO": s.gPO, "gCU": s.gCU, "gOBJ": s.gOBJ, "gFLT": s.gFLT}
        except Exception:  # noqa: BLE001
            pass

        row = {
            "i": i,
            "t": stamp,
            "joints": joints,
            "wrench": wrench,
            "suction_cmd": False,          # format compat with case_pick takes
            "vacuum_sealed": None,
            "barcode_confirmed": False,
            "gripper_pos": None if grip is None else grip["gPO"],
            "gripper": grip,
            "gripper_cmd": dict(self.gripper_cmd),
            "ee": {"frame": ikcfg.GRIPPER_EE_FRAME, "pos": [float(v) for v in pos],
                   "quat_wxyz": _rpy_to_quat_wxyz(rpy)},
        }
        cur["states"].write(json.dumps(row) + "\n")
        cur["idx"] = i + 1


# ---------------------------------------------------------------------------
# Keyboard jog (raw-key, same bindings as case_battery_demo.teach_pose)
# ---------------------------------------------------------------------------
_POS_KEYS = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1), "r": (2, +1), "f": (2, -1)}
_RPY_KEYS = {"u": (0, +1), "o": (0, -1), "i": (1, +1), "k": (1, -1), "j": (2, +1), "l": (2, -1)}


def _get_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _jog(mover: ArmMover, start_pos, start_rpy) -> tuple[np.ndarray, np.ndarray] | None:
    """Jog the EE from (start_pos, start_rpy); ENTER returns the frozen grasp
    pose, q/Ctrl-C returns None. Tracks the COMMANDED target (not live FK) so
    the grasp pose is exactly what was steered, free of droop/solve error."""
    target_pos = np.asarray(start_pos, dtype=float).copy()
    target_rpy = np.asarray(start_rpy, dtype=float).copy()
    step = float(rcfg.JOG_STEP_M)
    ostep = float(rcfg.JOG_OSTEP_DEG)

    def _status() -> str:
        return (f"pos=({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})  "
                f"rpy_deg=({np.rad2deg(target_rpy[0]):.1f}, {np.rad2deg(target_rpy[1]):.1f}, "
                f"{np.rad2deg(target_rpy[2]):.1f})  | step={step * 100:.2f}cm ostep={ostep:.1f}deg")

    print("[jog] w/s=+x/-x a/d=+y/-y r/f=+z/-z · u/o i/k j/l=rpy · +/-=step · "
          "ENTER=grasp here · q=abort")
    print(_status())
    while True:
        key = _get_key()
        if key in ("\r", "\n"):
            return target_pos, target_rpy
        if key == "q" or ord(key) == 3:  # q or Ctrl+C
            return None
        if key in _POS_KEYS:
            axis, sign = _POS_KEYS[key]
            target_pos[axis] += sign * step
            if mover.move_ee(target_pos, target_rpy) is None:
                target_pos[axis] -= sign * step  # unreachable — undo
                logger.warning("[jog] step unreachable — reverted")
        elif key in _RPY_KEYS:
            axis, sign = _RPY_KEYS[key]
            target_rpy[axis] += sign * np.deg2rad(ostep)
            if mover.move_ee(target_pos, target_rpy) is None:
                target_rpy[axis] -= sign * np.deg2rad(ostep)
                logger.warning("[jog] step unreachable — reverted")
        elif key in ("+", "="):
            step = min(step * 2, rcfg.JOG_STEP_MAX_M)
        elif key == "-":
            step = max(step / 2, rcfg.JOG_STEP_MIN_M)
        elif key in (",", "."):
            ostep = max(1.0, ostep / 2) if key == "," else min(20.0, ostep * 2)
        else:
            continue
        print(_status())


def _open_elbow(mover: ArmMover, pos, rpy, q0: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Walk the elbow-swivel null space at a fixed EE pose, pushing j2 down
    (right arm: negative j2 = elbow out / armpit open) until IK stops
    improving or validity breaks. Each re-solve pins the posture target to
    the pushed seed, so the QP returns the EE-consistent config nearest the
    more-open elbow. Planning-time only (a few dozen warm solves)."""
    q = np.asarray(q0, dtype=float).copy()
    for _ in range(100):
        seed = q.copy()
        seed[1] -= step  # below-limit seeds get clipped inside solve_pose
        sol = mover.solve_pose(pos, rpy, seed=seed, min_motion=True)
        if not sol.valid or sol.q[1] >= q[1] - 1e-4:
            break
        q = sol.q
    return q


def _move_to_standoff(mover: ArmMover, pos, rpy) -> tuple[np.ndarray, float] | None:
    """Move to the deepest reachable standoff of (pos, rpy): back off along
    the tool +z by STANDOFF_M, shrinking 5 cm at a time down to STANDOFF_MIN_M
    when the depth leaves the workspace (deep retreats from a forward pose end
    up too close to the torso — observed 54 mm short at 0.35 m). A failed
    move_ee does not move the arm, so only the first reachable depth moves.
    Returns (standoff_point, depth) or None if no depth is reachable."""
    approach = Rotation.from_euler("xyz", rpy).as_matrix()[:, 2]  # tool +z in base_link
    depth = float(rcfg.STANDOFF_M)
    while depth >= rcfg.STANDOFF_MIN_M - 1e-9:
        cand = np.asarray(pos, dtype=float) - depth * approach
        if mover.move_ee(cand, rpy) is not None:
            if depth < rcfg.STANDOFF_M - 1e-9:
                logger.warning("standoff shrunk to {:.2f} m (configured {:.2f} m unreachable)",
                               depth, rcfg.STANDOFF_M)
            return cand, depth
        depth -= 0.05
    logger.error("no reachable standoff between {:.2f} and {:.2f} m",
                 rcfg.STANDOFF_M, rcfg.STANDOFF_MIN_M)
    return None


def _sample_place(mover: ArmMover, z: float, rpy, lift_z: float) -> np.ndarray | None:
    """Uniform-random put-down position inside the PLACE box, IK-checked (the
    transfer waypoint at lift height AND the put-down pose must both solve).
    Returns [x, y, z] or None after PLACE_MAX_TRIES rejections."""
    seed = mover._live_arm_q()
    for _ in range(int(rcfg.PLACE_MAX_TRIES)):
        x = float(np.random.uniform(*rcfg.PLACE_X_RANGE))
        y = float(np.random.uniform(*rcfg.PLACE_Y_RANGE))
        s1 = mover.solve_pose([x, y, lift_z], rpy, seed=seed, min_motion=True)
        if not s1.valid:
            continue
        s2 = mover.solve_pose([x, y, z], rpy, seed=s1.q, min_motion=True)
        if s2.valid:
            return np.array([x, y, z], dtype=float)
    return None


def _release_to_contact(mover: ArmMover, floor_z: float, rpy) -> bool | None:
    """Creep the HELD object straight down until the right wrist F/T sensor
    reports table contact (tared base-vertical force > RELEASE_CONTACT_N) —
    the release analog of suction._descend_to_contact. Tare happens here, so
    call at the start height with the object free of the table. Returns True
    on contact, False on stall / floor_z reached without contact (caller just
    opens where it stopped), None when the sensor is unavailable (caller
    falls back to the fixed-height drop)."""
    ws = getattr(mover._arm, "wrench_sensor", None)
    if ws is None:
        return None
    try:
        samples = []
        for _ in range(20):
            samples.append(np.asarray(ws.get_wrench_state()[:3], dtype=float))
            time.sleep(0.005)
        baseline = np.mean(samples, axis=0)
    except Exception as e:  # noqa: BLE001
        logger.warning("[release] wrench read failed ({}) — fixed-height drop", e)
        return None

    def _vertical_n() -> float:
        raw = np.asarray(ws.get_wrench_state()[:3], dtype=float) - baseline
        return float(abs((mover.current_ee_rotation() @ raw)[2]))

    dt = 1.0 / float(ikcfg.CONTROL_HZ)
    prev_q = mover._live_arm_q()
    pos, _ = mover.fk(prev_q)
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    elapsed = 0.0
    logger.info("[release] creep from z={:.4f} to contact (floor {:.4f})", z, floor_z)
    while z > floor_z:
        r = min(1.0, elapsed / 0.3)
        z_next = z - rcfg.RELEASE_SPEED_M_S * (r * r * (3.0 - 2.0 * r)) * dt
        sol = mover.solve_pose([x, y, z_next], rpy, seed=prev_q, min_motion=True)
        if sol.pos_err_m > ikcfg.REACH_TOL_M:
            mover._arm.set_joint_pos_vel(prev_q, np.zeros(7))
            logger.warning("[release] descent stalled at z={:.4f} — opening here", z)
            return False
        mover._arm.set_joint_pos_vel(sol.q, (sol.q - prev_q) / dt)
        z, prev_q = z_next, sol.q
        try:
            f = _vertical_n()
        except Exception:  # noqa: BLE001
            f = 0.0
        if f > rcfg.RELEASE_CONTACT_N:
            mover._arm.set_joint_pos_vel(sol.q, np.zeros(7))
            logger.info("[release] table contact {:.1f}N at z={:.4f}", f, z)
            return True
        elapsed += dt
        time.sleep(dt)
    mover._arm.set_joint_pos_vel(prev_q, np.zeros(7))
    logger.warning("[release] no contact down to z={:.4f} — opening here", floor_z)
    return False


def _grip_soft(gripper: RobotiqGripperUSB, rec: EpisodeRecorder, force: int) -> bool:
    """Contact-stop close: stream a slow full-close command (non-blocking),
    poll the status, and at the first contact signature freeze the position
    target at the current gPO (+ SOFT_GRIP_SQUEEZE counts). The Robotiq force
    controller can't go below ~20 N even at force 0; holding a *position*
    just past contact applies only the elastic squeeze, so the effective
    grip force is far lower, bounded by detection latency. Contact = gCU >=
    SOFT_GRIP_CU_STOP (free-travel noise measured <= ~7 counts); the first
    150 ms are ignored (motor inrush). Returns True if an object is held."""
    speed = int(rcfg.GRIP_SPEED)
    rec.gripper_cmd.update(pos=ikcfg.ROBOTIQ_CLOSE_POS, speed=speed, force=int(force))
    gripper._write_control(0x09, ikcfg.ROBOTIQ_CLOSE_POS, speed, int(force))
    t0 = time.time()
    cu_hits = 0
    while time.time() - t0 < rcfg.SOFT_GRIP_TIMEOUT_S:
        s = gripper.read_status()
        if s is None:
            time.sleep(0.02)
            continue
        if s.gOBJ == 3:
            return False  # fully closed — nothing met the fingers
        if s.gOBJ == 2:
            logger.info("[soft-grip] force controller stalled first (gPO={} gCU={})",
                        s.gPO, s.gCU)
            return True
        # Consecutive polls above the threshold: single-poll noise and brief
        # one-finger grazes don't freeze — the close keeps pushing the object
        # toward the other finger (self-centering) until BOTH fingers load.
        if time.time() - t0 > 0.15 and s.gCU >= rcfg.SOFT_GRIP_CU_STOP:
            cu_hits += 1
        else:
            cu_hits = 0
        if cu_hits >= int(rcfg.SOFT_GRIP_CU_CONSECUTIVE):
            target = min(255, s.gPO + int(rcfg.SOFT_GRIP_SQUEEZE))
            gripper._write_control(0x09, target, speed, int(force))
            rec.gripper_cmd.update(pos=target, speed=speed, force=int(force))
            logger.info("[soft-grip] contact at gPO={} (gCU={} ~{}mA, {} polls) — frozen at {}",
                        s.gPO, s.gCU, s.gCU * 10, cu_hits, target)
            time.sleep(0.3)  # settle; then judge by the stopped-short gap
            return gripper.is_object_grasped()
        time.sleep(0.02)
    logger.warning("[soft-grip] no contact within {:.1f}s", rcfg.SOFT_GRIP_TIMEOUT_S)
    return gripper.is_object_grasped()


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def _home_via_lift(mover: ArmMover) -> None:
    """Project-local home return (used everywhere, incl. session start/end):
    straight-vertical lift to min(home_z, ee_z + HOME_LIFT_DZ_M), then the
    joint move home. The objects here are small, so +20 cm of clearance beats
    ik_demo safe_home's absolute 1.05 m lift; capping at the home height keeps
    it from overshooting home and coming back down, and skips the lift
    entirely when the EE is already at/above it."""
    pos, rpy = mover.current_ee_pose()
    home_z = float(mover.fk(mover._home_seed)[0][2])
    target_z = min(home_z, float(pos[2]) + rcfg.HOME_LIFT_DZ_M)
    if target_z > float(pos[2]) + 0.01:
        if mover.move_ee_vertical(target_z, tuple(rpy)) is None:
            logger.warning("[home] lift stalled — homing from the current pose")
    mover.move_joints(mover._home_seed)


def _cycle(mover: ArmMover, gripper: RobotiqGripperUSB, rec: EpisodeRecorder,
           force: int, obj: str,
           grasp_hint: tuple[np.ndarray, np.ndarray] | None = None,
           monitor: CollisionMonitor | None = None,
           ) -> tuple[np.ndarray, np.ndarray] | None:
    """One episode. With grasp_hint (pos, rpy) — the previous cycle's random
    put-down spot — the default-pose visit and the keyboard jog are skipped
    and the pick starts straight from home. Returns the hint for the NEXT
    cycle (this cycle's put-down position), or None when the object's
    location is unknown (pick failed / dropped) so the next cycle re-jogs."""
    # 1. gripper opens first, before the arm gets anywhere near the object
    rec.gripper_cmd.update(pos=ikcfg.ROBOTIQ_OPEN_POS, speed=ikcfg.ROBOTIQ_SPEED,
                           force=ikcfg.ROBOTIQ_FORCE)
    gripper.open()

    if grasp_hint is not None:
        # 2a. object location known from the last random put-down — no jog
        grasp_pos, grasp_rpy = grasp_hint
        logger.info("auto grasp target (last put-down): ({:.3f}, {:.3f}, {:.3f})", *grasp_pos)
    else:
        # 2b. default grip pose via its standoff (approach from behind along
        # the tool axis — never sweep the object area), the user jogs to the
        # object, ENTER freezes the grasp pose; retreat + home.
        if rcfg.DEFAULT_GRIP_POSE is not None:
            pose = np.asarray(rcfg.DEFAULT_GRIP_POSE, dtype=float)
            start_pos, start_rpy = pose[:3], pose[3:6]
            if _move_to_standoff(mover, start_pos, start_rpy) is None:
                logger.error("cannot approach DEFAULT_GRIP_POSE — aborting cycle")
                return None
            if mover.move_ee(start_pos, start_rpy) is None:
                logger.error("DEFAULT_GRIP_POSE unreachable — fix right_gripper/config.py")
                return None
        else:
            logger.warning("DEFAULT_GRIP_POSE not set — jogging from the current pose "
                           "(teach it and paste into right_gripper/config.py)")
            start_pos, start_rpy = mover.current_ee_pose()
        res = _jog(mover, start_pos, start_rpy)
        if res is None:
            logger.info("cycle aborted")
            return None
        grasp_pos, grasp_rpy = res
        if _move_to_standoff(mover, grasp_pos, grasp_rpy) is None:
            logger.error("standoff unreachable — aborting cycle")
            return None
        _home_via_lift(mover)

    # 3. recorded pick: home -> standoff -> in -> close -> lift -> hold
    rec.begin(
        instruction=rcfg.INSTRUCTION_TMPL.format(obj=obj),
        extra={
            "object": obj,
            "grip_force": int(force),
            "grip_speed": int(rcfg.GRIP_SPEED),
            "grasp_pos": [float(v) for v in grasp_pos],
            "grasp_rpy": [float(v) for v in grasp_rpy],
            "grasp_source": "auto" if grasp_hint is not None else "jog",
            "grip_mode": "soft" if rcfg.SOFT_GRIP else "force",
            "lift_dz_m": float(rcfg.LIFT_DZ_M),
        },
    )
    grasped = lifted = held = False
    reason = "ok"
    standoff_depth = None
    res = _move_to_standoff(mover, grasp_pos, grasp_rpy)
    if res is None or mover.move_ee(grasp_pos, grasp_rpy) is None:
        reason = "approach_unreachable"
        logger.error("approach unreachable — ending take")
    else:
        standoff_depth = res[1]
        if monitor is not None:  # intentional contact: grip reaction is not a collision
            monitor.suppress(True)
        if rcfg.SOFT_GRIP:
            grasped = _grip_soft(gripper, rec, force)
        else:
            rec.gripper_cmd.update(pos=ikcfg.ROBOTIQ_CLOSE_POS, speed=rcfg.GRIP_SPEED,
                                   force=int(force))
            gripper.goto(ikcfg.ROBOTIQ_CLOSE_POS, speed=rcfg.GRIP_SPEED, force=int(force))
            grasped = gripper.is_object_grasped()
        if monitor is not None:
            monitor.suppress(False)
            if grasped and rcfg.OBJECT_MASS_KG > 0:
                monitor.set_extra_payload(rcfg.OBJECT_MASS_KG)
        logger.info("close -> {}", "GRIPPED" if grasped else "no object")
        lifted = mover.move_ee_vertical(float(grasp_pos[2]) + rcfg.LIFT_DZ_M, grasp_rpy) is not None
        time.sleep(rcfg.HOLD_S)
        held = gripper.is_object_grasped()
        if not grasped:
            reason = "no_object_on_close"
        elif not lifted:
            reason = "lift_stalled"
        elif not held:
            reason = "dropped_during_hold"
    collided = bool(monitor is not None and monitor.triggered)
    rec.end(success=grasped and lifted and held and not collided,
            extra={"grasped_on_close": grasped, "lifted": lifted,
                   "held_after_hold": held, "pick_reason": reason,
                   "collision": collided,
                   "standoff_m": None if standoff_depth is None else float(standoff_depth)})
    if collided:
        logger.error("[collision] cycle aborted — arm holds its pose (stream gated); "
                     "the object may still be in the gripper. 'r' at the prompt re-arms.")
        return None

    # 4. put down at a random spot inside the PLACE box (that spot becomes the
    # next episode's grasp target); falls back to the pick spot when sampling
    # or the transfer fails. Not recorded — the take closed at the hold.
    place_pos = None
    lift_z = float(grasp_pos[2]) + rcfg.LIFT_DZ_M
    if held:
        place_pos = _sample_place(mover, z=float(grasp_pos[2]), rpy=grasp_rpy, lift_z=lift_z)
        if place_pos is None:
            logger.warning("no reachable random put-down — placing back at the pick spot")
            place_pos = np.asarray(grasp_pos, dtype=float).copy()
        elif mover.move_ee([place_pos[0], place_pos[1], lift_z], grasp_rpy) is None:
            logger.warning("transfer unreachable — placing back at the pick spot")
            place_pos = np.asarray(grasp_pos, dtype=float).copy()
        else:
            logger.info("put-down at ({:.3f}, {:.3f})", place_pos[0], place_pos[1])
    release_at = np.asarray(grasp_pos if place_pos is None else place_pos, dtype=float)
    contact = None
    if lifted:
        # Descend to the re-align height, fix the FULL pose there (xy + rpy
        # drift from the endpoint-only transfer — a tilted release tips the
        # object), then set the object down on the wrist F/T contact signal
        # instead of dropping it from a fixed height.
        start_z = float(grasp_pos[2]) + rcfg.RELEASE_START_DZ_M
        mover.move_ee_vertical(start_z, grasp_rpy)
        mover.move_ee([float(release_at[0]), float(release_at[1]), start_z], grasp_rpy)
        if rcfg.RELEASE_ON_CONTACT:
            if monitor is not None:  # intentional contact: setting the object down.
                monitor.suppress(True)  # resumed below, AFTER open releases the force
            contact = _release_to_contact(
                mover, float(grasp_pos[2]) - rcfg.RELEASE_FLOOR_DZ_M, grasp_rpy)
        if contact is None:  # disabled or sensor unavailable — legacy fixed drop
            mover.move_ee_vertical(float(grasp_pos[2] + 0.02), grasp_rpy)
    if monitor is not None and monitor.triggered:
        # Collision latched mid-put-down: the arm holds its pose (stream
        # gated) — do NOT open, the object may be hanging above the table.
        monitor.suppress(False)
        if rcfg.OBJECT_MASS_KG > 0:
            monitor.set_extra_payload(0.0)
        logger.error("[collision] latched during put-down — arm holds pose, gripper NOT "
                     "opened (object still held). 'r' at the prompt re-arms.")
        return None
    rec.gripper_cmd.update(pos=ikcfg.ROBOTIQ_OPEN_POS, speed=ikcfg.ROBOTIQ_SPEED,
                           force=ikcfg.ROBOTIQ_FORCE)
    gripper.open()
    if monitor is not None:  # contact force is gone now that the fingers opened
        monitor.suppress(False)
        if rcfg.OBJECT_MASS_KG > 0:
            monitor.set_extra_payload(0.0)
    _move_to_standoff(mover, release_at, grasp_rpy)  # back off; on failure home directly
    _home_via_lift(mover)
    if place_pos is not None:
        rec.amend_last({"place_pos": [float(v) for v in place_pos],
                        "release_contact": contact})

    # 5. human label — the take is closed, so relabel/discard in place
    ans = input("[label] object intact & pick good? y=success n=fail d=discard ENTER=keep auto > ").strip().lower()
    if ans == "y":
        rec.relabel_last(success=True, extra={"human_label": "success"})
    elif ans == "n":
        rec.relabel_last(success=False, extra={"human_label": "fail"})
    elif ans == "d":
        rec.discard_last()

    if place_pos is None:
        return None
    return (np.array([place_pos[0], place_pos[1], float(grasp_pos[2])]),
            np.asarray(grasp_rpy, dtype=float).copy())


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _main() -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    if rcfg.DEFAULT_GRIP_POSE is not None and len(rcfg.DEFAULT_GRIP_POSE) != 6:
        logger.error("DEFAULT_GRIP_POSE must be 6 values (x, y, z, roll, pitch, yaw) — "
                     "got {}: {}", len(rcfg.DEFAULT_GRIP_POSE), rcfg.DEFAULT_GRIP_POSE)
        return

    logger.warning("=" * 60)
    logger.warning("MOVES THE REAL RIGHT ARM + GRIPPER. Each cycle: default pose ->")
    logger.warning("keyboard jog -> ENTER -> home -> recorded pick + lift -> put back.")
    logger.warning("E-stop in reach. Set grip force with 'f N' between episodes.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        try:
            sys.path.insert(0, str(_REPO_ROOT / "perception"))
            from utils import set_head_pitch  # noqa: PLC0415
            set_head_pitch(bot, angle=30.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("set_head_pitch unavailable ({}) — aim the head manually", e)

        mover = ArmMover(bot, side="right", ee_frame=ikcfg.GRIPPER_EE_FRAME)
        release = mover.software_estop_active()
        if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
            return
        if not mover.ensure_ready(release_estop=release):
            logger.error("arm not ready — aborting")
            return

        # Tighten R_arm_j2's upper bound on this mover's model: the IK QP
        # limit constraint, seed clipping, and in_limits validation all read
        # from it, so every solve this session keeps the elbow from folding
        # inward. AFTER ensure_ready (pin_torso rebuilds the model) and before
        # the home derivation so home respects the cap too.
        if rcfg.J2_MAX is not None:
            iq = mover._model.idx_qs[mover._arm_joint_ids[1]]
            mover._model.upperPositionLimit[iq] = float(rcfg.J2_MAX)
            mover._q_hi = mover._model.upperPositionLimit.copy()
            mover._q_lo = mover._model.lowerPositionLimit.copy()
            mover._setup_ik()  # rebuild limit constraints + posture mid on the tightened box
            logger.info("R_arm_j2 capped at {:+.2f} rad (URDF upper +0.45)", rcfg.J2_MAX)

        # Project-local home. Must be set AFTER ensure_ready: its pin_torso
        # rebuilds the model and resets _home_seed to the ik_demo home.
        # _home_via_lift + IK seeding then use this instead.
        if rcfg.HOME_JOINTS is not None:
            home_q = np.asarray(rcfg.HOME_JOINTS, dtype=float)
            if home_q.shape != (7,) or not mover.in_limits(home_q) or mover.in_collision(home_q):
                logger.error("right_gripper HOME_JOINTS invalid (needs 7 in-limit, "
                             "collision-free joints) — fix right_gripper/config.py")
                return
            mover._home_seed = home_q
            # Snap the home ORIENTATION to the grasp rpy, keeping HOME_JOINTS'
            # position: solve IK once here so every home return starts with the
            # wrist already grasp-aligned (episodes then approach by translation,
            # no big mid-approach reorientation).
            if rcfg.HOME_MATCH_GRIP_RPY and rcfg.DEFAULT_GRIP_POSE is not None:
                home_pos = (np.asarray(rcfg.HOME_POS, dtype=float)
                            if rcfg.HOME_POS is not None else mover.fk(home_q)[0])
                grip_rpy = np.asarray(rcfg.DEFAULT_GRIP_POSE, dtype=float)[3:6]
                sol = mover.solve_pose(home_pos, grip_rpy, seed=home_q, min_motion=True)
                if not sol.valid:
                    logger.error("home @ grip rpy unsolvable (err={:.1f}mm converged={} "
                                 "collision={} in_limits={}) — adjust HOME_JOINTS or set "
                                 "HOME_MATCH_GRIP_RPY=False", sol.pos_err_m * 1000,
                                 sol.converged, sol.in_collision, sol.in_limits)
                    return
                home_sol_q = sol.q
                if rcfg.HOME_OPEN_ELBOW:
                    home_sol_q = _open_elbow(mover, home_pos, grip_rpy, home_sol_q)
                mover._home_seed = home_sol_q
                logger.info("home @ grip rpy — pos {} kept, j2={:+.3f} (elbow-open), q={}",
                            np.round(home_pos, 3), home_sol_q[1], np.round(home_sol_q, 3))

        gripper = RobotiqGripperUSB()
        if not gripper.initialize():
            logger.error("gripper unavailable — aborting")
            return
        gripper.open()

        logger.info("-> right arm home")
        _home_via_lift(mover)

        # Collision watchdog (model-based, Collision/): on trigger the monitor
        # freezes both arms at their current pose. NO E-Stop — the estop
        # service is dead on this firmware (soc 429 < min 1200; every call
        # times out). Instead OUR OWN command stream is gated: while the
        # collision is latched, set_joint_pos_vel ticks are dropped, so
        # in-flight legs can't fight the freeze. The arm simply holds pose
        # until 'r' at the prompt re-arms.
        try:
            with open(_REPO_ROOT / "Collision" / "calibration_right.json") as f:
                _calib = json.load(f)
            colmon = CollisionMonitor(
                bot, side="right",
                thresholds=rcfg.COLLISION_ABS_SCALE
                * np.asarray(_calib["banded_suggested_thresholds"], dtype=float),
                change_thresholds=rcfg.COLLISION_CHG_SCALE
                * np.asarray(_calib["change_suggested_thresholds"], dtype=float),
            )
        except FileNotFoundError:
            logger.error("Collision/calibration_right.json not found — run: "
                         "python Collision/calibrate_gravity_model.py --side right")
            if input("Continue WITHOUT collision monitoring? [y/N]: ").strip().lower() != "y":
                return
            colmon = None
        if colmon is not None:
            colmon.start()
            _raw_sjpv = mover._arm.set_joint_pos_vel

            def _guarded_sjpv(*a, **k):
                if colmon.triggered:
                    return None  # collision latched — hold pose, drop stream ticks
                return _raw_sjpv(*a, **k)

            mover._arm.set_joint_pos_vel = _guarded_sjpv

        rec = EpisodeRecorder(bot, gripper, OUT_DIR).start()
        force = int(rcfg.GRIP_FORCE)
        obj = rcfg.OBJECT_DEFAULT
        hint: tuple[np.ndarray, np.ndarray] | None = None  # next grasp = last put-down
        try:
            while True:
                nxt = (f"auto({hint[0][0]:.2f},{hint[0][1]:.2f})" if hint is not None
                       else "jog")
                collided = colmon is not None and colmon.triggered
                state = "COLLISION(r=re-arm)" if collided else nxt
                cmd = input(f"[collect] takes={rec.kept} force={force} obj={obj} "
                            f"next={state} — ENTER=episode · j=re-jog · f N=force · "
                            "o NAME=object · r=re-arm · d=discard last · q=quit > ").strip()
                if cmd.lower() == "q":
                    break
                if cmd.lower() == "r":
                    if colmon is not None:
                        colmon.reset()  # un-latch FIRST: reopens the stream gate
                    logger.info("re-armed — returning home")
                    _home_via_lift(mover)
                    hint = None  # object state unknown after a collision
                    continue
                if cmd.lower() == "j":
                    hint = None
                    continue
                if cmd.lower() == "d":
                    rec.discard_last()
                    continue
                if cmd.lower().startswith("f"):
                    try:
                        force = max(0, min(255, int(cmd.split()[1])))
                    except (IndexError, ValueError):
                        logger.warning("usage: f N   (grip force 0..255)")
                    continue
                if cmd.lower().startswith("o"):
                    try:
                        obj = cmd.split(maxsplit=1)[1].strip()
                    except IndexError:
                        logger.warning("usage: o NAME   (object name for the instruction)")
                    continue
                if cmd == "":
                    if collided:
                        logger.warning("collision latched — press 'r' to re-arm first")
                        continue
                    hint = _cycle(mover, gripper, rec, force, obj, grasp_hint=hint,
                                  monitor=colmon)
        finally:
            if colmon is not None:
                colmon.stop()
            rec.stop()
        logger.info("-> home ({} takes kept)", rec.kept)
        if colmon is not None and colmon.triggered:
            logger.warning("collision still latched — leaving the arm where it froze")
        else:
            _home_via_lift(mover)


if __name__ == "__main__":
    _main()
