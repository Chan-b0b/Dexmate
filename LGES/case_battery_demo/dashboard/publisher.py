"""Background publisher that spools live robot state for the dashboard viewer.

Runs on a daemon thread inside the demo process (the only place that safely
holds the Robot connection — a second Robot() would re-home the head and reset
arm modes, fighting the running demo). Each tick it samples:

    * the head camera's left RGB frame      -> <spool>/frame.jpg
    * joints / EE pose / wrench snapshot     -> <spool>/state.json

Both files are written atomically (tmp + os.replace) so the viewer never reads
a half-written file. The viewer (case_battery_demo.dashboard.server) polls the
spool, so it stays fully decoupled from this process.

EE pose is computed here with its own pinocchio model/data — the SuctionMover's
model is NOT thread-safe, so we never call into it from this thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import pinocchio as pin
from loguru import logger
from scipy.spatial.transform import Rotation

from .. import config as cfg

# read_force lives in grasp_box/ (same path grasp.py wires up). Importing it
# here shares the module-level tare baseline the demo sets during pick/place,
# so the magnitude we show matches what the grasp loop thresholds on.
_GRASP_BOX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "grasp_box")
)
if _GRASP_BOX_DIR not in sys.path:
    sys.path.insert(0, _GRASP_BOX_DIR)
try:
    from read_force import get_force  # noqa: E402
except Exception:  # noqa: BLE001 - fall back to raw wrench magnitude
    get_force = None

DEFAULT_SPOOL_DIR = "/tmp/cns_dashboard"


class _EEKinematics:
    """Forward kinematics for the suction arm's EE frame, in base_link.

    Uses the full URDF model with the live torso + arm joints set; the EE frame
    (cfg.EE_FRAME) only depends on that chain, so the rest stay at neutral.
    Owns its own pinocchio data, independent of the SuctionMover.
    """

    def __init__(self) -> None:
        urdf = cfg.URDF_PATH
        rw = pin.RobotWrapper.BuildFromURDF(
            filename=urdf,
            package_dirs=[
                os.path.dirname(urdf),
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(urdf)))),
            ],
            root_joint=None,
        )
        self._model = rw.model
        self._data = self._model.createData()
        self._ee_fid = self._model.getFrameId(cfg.EE_FRAME)
        prefix = "R" if cfg.ARM_SIDE == "right" else "L"
        self._arm_jids = [self._model.getJointId(f"{prefix}_arm_j{i + 1}") for i in range(7)]
        self._torso_jids = [self._model.getJointId(f"torso_j{i + 1}") for i in range(3)]

    def compute(self, torso_q, arm_q) -> tuple[np.ndarray, np.ndarray]:
        """Return (position [x,y,z] m, rpy [r,p,y] rad) of the EE in base_link."""
        q = pin.neutral(self._model)
        for jid, v in zip(self._torso_jids, np.asarray(torso_q, dtype=float)):
            q[self._model.idx_qs[jid]] = v
        for jid, v in zip(self._arm_jids, np.asarray(arm_q, dtype=float)):
            q[self._model.idx_qs[jid]] = v
        pin.framesForwardKinematics(self._model, self._data, q)
        T = self._data.oMf[self._ee_fid]
        rpy = Rotation.from_matrix(T.rotation).as_euler("xyz")
        return T.translation.copy(), rpy


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class DashboardPublisher:
    """Samples robot state on a daemon thread and spools it for the viewer."""

    def __init__(
        self,
        robot,
        spool_dir: str = DEFAULT_SPOOL_DIR,
        hz: float = 15.0,
        max_image_width: int = 720,
        jpeg_quality: int = 80,
        depth_range_m: tuple[float, float] = (0.3, 1.0),
    ) -> None:
        self._robot = robot
        self.spool_dir = spool_dir
        self._period = 1.0 / max(hz, 1.0)
        self._max_w = max_image_width
        self._jpeg_quality = int(jpeg_quality)
        self._frame_path = os.path.join(spool_dir, "frame.jpg")
        self._depth_path = os.path.join(spool_dir, "depth.jpg")
        self._state_path = os.path.join(spool_dir, "state.json")
        self._depth_range = (float(depth_range_m[0]), float(depth_range_m[1]))
        self._depth_cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0
        os.makedirs(spool_dir, exist_ok=True)
        try:
            self._fk: _EEKinematics | None = _EEKinematics()
        except Exception as e:  # noqa: BLE001
            logger.warning("[dashboard] EE kinematics unavailable: {}", e)
            self._fk = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "DashboardPublisher":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="dashboard-publisher", daemon=True
        )
        self._thread.start()
        logger.info(
            "[dashboard] publishing to {} — start the viewer with:\n"
            "    python -m case_battery_demo.dashboard.server --spool {}",
            self.spool_dir, self.spool_dir,
        )
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "DashboardPublisher":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- sampling ----------------------------------------------------------

    @property
    def _arm(self):
        return self._robot.left_arm if cfg.ARM_SIDE == "left" else self._robot.right_arm

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._sample_once()
            except Exception as e:  # noqa: BLE001 - never let one bad read kill the thread
                logger.debug("[dashboard] sample error: {}", e)
            dt = time.time() - t0
            self._stop.wait(max(0.0, self._period - dt))

    def _sample_once(self) -> None:
        self._seq += 1
        state: dict = {"seq": self._seq, "stamp": time.time(), "arm_side": cfg.ARM_SIDE}

        # --- head camera: left RGB + depth -> frame.jpg / depth.jpg -------
        has_image = has_depth = False
        try:
            if self._robot.has_sensor("head_camera"):
                cam = self._robot.sensors.head_camera
                rgb = cam.get_left_rgb()
                if rgb is not None:
                    has_image = self._write_frame(rgb)
                depth = cam.get_depth()
                if depth is not None:
                    has_depth = self._write_depth(depth)
        except Exception as e:  # noqa: BLE001
            logger.debug("[dashboard] camera read error: {}", e)
        state["has_image"] = has_image
        state["has_depth"] = has_depth
        state["depth_range_m"] = list(self._depth_range)

        # --- joints (rad) -------------------------------------------------
        joints: dict[str, dict[str, float]] = {}
        for comp in ("left_arm", "right_arm", "torso", "head"):
            try:
                if self._robot.has_component(comp):
                    jp = getattr(self._robot, comp).get_joint_pos_dict()
                    joints[comp] = {k: float(v) for k, v in jp.items()}
            except Exception:  # noqa: BLE001
                pass
        state["joints"] = joints

        # --- EE pose (base_link) ------------------------------------------
        if self._fk is not None:
            try:
                torso_q = self._robot.torso.get_joint_pos()
                arm_q = self._arm.get_joint_pos()
                pos, rpy = self._fk.compute(torso_q, arm_q)
                state["ee"] = {
                    "frame": cfg.EE_FRAME,
                    "pos": [float(v) for v in pos],
                    "rpy": [float(v) for v in rpy],
                }
            except Exception as e:  # noqa: BLE001
                logger.debug("[dashboard] FK error: {}", e)

        # --- wrench (raw 6-axis + tared magnitude) ------------------------
        state["wrench"] = self._read_wrench()

        _atomic_write(self._state_path, json.dumps(state).encode("utf-8"))

    def _write_frame(self, rgb: np.ndarray) -> bool:
        # head camera returns RGB; cv2 encodes assuming BGR, so convert first.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if w > self._max_w:
            scale = self._max_w / float(w)
            bgr = cv2.resize(bgr, (self._max_w, int(round(h * scale))))
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return False
        _atomic_write(self._frame_path, buf.tobytes())
        return True

    def _write_depth(self, depth: np.ndarray) -> bool:
        color = self._colorize_depth(depth)
        h, w = color.shape[:2]
        if w > self._max_w:
            scale = self._max_w / float(w)
            color = cv2.resize(color, (self._max_w, int(round(h * scale))))
        ok, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return False
        _atomic_write(self._depth_path, buf.tobytes())
        return True

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Map a float32 depth map (metres) to a BGR colour image for display.

        Distances are clipped to the configured [near, far] range and run
        through a colormap; invalid pixels (NaN/inf/<=0) are painted black.
        applyColorMap returns BGR, so the result is encoded as-is (no swap).
        """
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:  # some streams hand back (H, W, 1)
            d = d[..., 0]
        near, far = self._depth_range
        valid = np.isfinite(d) & (d > 0.0)
        norm = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
        norm[~valid] = 0.0
        u8 = (norm * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(u8, self._depth_cmap)
        color[~valid] = (0, 0, 0)
        return color

    def _read_wrench(self) -> dict | None:
        ws = getattr(self._arm, "wrench_sensor", None)
        if ws is None:
            return None
        try:
            raw = np.asarray(ws.get_state()["wrench"], dtype=float)
        except Exception:  # noqa: BLE001 - ServiceUnavailableError etc.
            return None
        fx, fy, fz, tx, ty, tz = (float(v) for v in raw[:6])
        out = {
            "fx": fx, "fy": fy, "fz": fz,
            "tx": tx, "ty": ty, "tz": tz,
            "raw_mag": float(np.linalg.norm(raw[:3])),
        }
        if get_force is not None:
            try:
                m = get_force(cfg.ARM_SIDE, self._robot)
                out["tared_mag"] = None if m is None else float(m)
            except Exception:  # noqa: BLE001
                out["tared_mag"] = None
        return out
