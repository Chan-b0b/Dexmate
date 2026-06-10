"""Episode recorder for VLA data collection (manual, keyboard/dashboard driven).

A recording "take" is one episode: you press start, the demo keeps running, you
press stop, then decide keep/discard. Frames are streamed straight to a temp dir
so "discard" is just a delete and there is no big RAM buffer.

The recorder is *fed* the DashboardPublisher's existing per-tick samples (RGB,
depth, joints, EE pose, wrench) — it never opens its own Robot/camera. Per the
agreed VLA design it logs raw observations only; the EE-delta + suction action
and per-frame phase label are derived offline (phase + commanded joints come
from joining each frame's timestamp against ``cfg.TRACE_PATH``).

Two controllers drive the same state machine, race-free, because every command
is funnelled onto one queue drained by a single worker thread:
  * keyboard (KeyListener) — SPACE toggle start/stop, y keep, n discard
  * dashboard buttons      — the server writes ``<spool>/record.cmd``; this polls it
The worker also writes ``<spool>/record.json`` so the dashboard can show state.

    IDLE --start--> RECORDING --stop--> DECIDING --keep|discard--> IDLE
DECIDING auto-keeps after DECIDE_TIMEOUT_S so an unanswered take is never lost.

On-disk per take (one episode):
    <out_dir>/<YYYYmmdd-HHMMSS_epNNNN>/
        meta.json                 # task / frame conventions / how to recover actions
        head_rgb/000000.jpg ...
        head_depth/000000.png     # 16-bit millimetres (0 = invalid), if depth present
        states.jsonl              # one line/frame: t, joints, ee(pos+quat), wrench, suction, gripper_pos
"""

from __future__ import annotations

import json
import os
import queue
import random
import shutil
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from .. import config as cfg
from .. import robotiq
from .. import suction_io

DEFAULT_SPOOL_DIR = "/tmp/cns_dashboard"
# Take names / meta timestamps follow Korean time (KST, UTC+9) regardless of the
# machine clock. The per-frame epoch 't' in states.jsonl stays time.time().
_KST = timezone(timedelta(hours=9))
DECIDE_TIMEOUT_S = 20.0          # auto-keep a finished take if no decision in time
_FRAME_QUEUE_MAX = 120           # ~8 s at 15 Hz before we start dropping frames
# depth_preview.jpg: one colorized sample per take (raw mm PNGs look black in a
# viewer; this is just for eyeballing). Same colormap/range as the dashboard.
_DEPTH_CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
_PREVIEW_DEPTH_RANGE_M = (0.3, 1.0)

IDLE, RECORDING, DECIDING = "idle", "recording", "deciding"


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _rpy_to_quat_wxyz(rpy) -> list[float]:
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return [float(w), float(x), float(y), float(z)]


class RecordController:
    """Manual episode recorder. Thread-safe via a single command/worker thread."""

    def __init__(
        self,
        out_dir: str = "recordings",
        spool_dir: str = DEFAULT_SPOOL_DIR,
        instruction: str = "",
        decide_timeout_s: float = DECIDE_TIMEOUT_S,
    ) -> None:
        self.out_dir = os.path.abspath(out_dir)
        self.spool_dir = os.path.abspath(spool_dir)
        self.instruction = instruction or "pick the part and place it in the target"
        self.decide_timeout_s = float(decide_timeout_s)

        self._pending_dir = os.path.join(self.out_dir, ".pending")
        self._cmd_path = os.path.join(self.spool_dir, "record.cmd")
        self._status_path = os.path.join(self.spool_dir, "record.json")

        self._frame_q: queue.Queue = queue.Queue(maxsize=_FRAME_QUEUE_MAX)
        self._cmd_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # worker-thread-only state (no locks needed)
        self._state = IDLE
        self._recording = False        # read by feed() on the publisher thread
        self._cur: dict | None = None   # active take: dirs, frame index, states file, t0
        self._ep_index = 0
        self._takes_saved = 0
        self._last_saved = ""
        self._dropped = 0
        self._decide_t0 = 0.0
        self._last_cmd_stamp = 0.0
        self._last_status = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "RecordController":
        if self._thread is not None:
            return self
        os.makedirs(self._pending_dir, exist_ok=True)
        # Ignore any stale command file left from a previous run.
        self._last_cmd_stamp = time.time()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        logger.info("[record] ready — takes -> {} (idle)", self.out_dir)
        return self

    def stop(self) -> None:
        # Keep an in-flight take rather than lose it on shutdown.
        if self._state == RECORDING:
            self.cmd_stop()
        if self._state in (RECORDING, DECIDING):
            self.cmd_keep()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "RecordController":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # -- public command API (callable from any thread) ---------------------

    def cmd_toggle(self) -> None:
        self._cmd_q.put("toggle")

    def cmd_start(self) -> None:
        self._cmd_q.put("start")

    def cmd_stop(self) -> None:
        self._cmd_q.put("stop")

    def cmd_keep(self) -> None:
        self._cmd_q.put("keep")

    def cmd_discard(self) -> None:
        self._cmd_q.put("discard")

    # -- sample sink (called on the publisher thread) ----------------------

    def feed(self, rgb, depth, state: dict) -> None:
        """Enqueue one observation. No-op unless a take is recording."""
        if not self._recording:
            return
        try:
            self._frame_q.put_nowait((
                None if rgb is None else rgb.copy(),
                None if depth is None else depth.copy(),
                state,
            ))
        except queue.Full:
            self._dropped += 1

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        self._write_status()
        while not self._stop.is_set():
            self._poll_cmd_file()
            while True:
                try:
                    self._apply(self._cmd_q.get_nowait())
                except queue.Empty:
                    break
            if self._state == DECIDING and time.time() - self._decide_t0 > self.decide_timeout_s:
                logger.info("[record] no decision in {:.0f}s — keeping take.", self.decide_timeout_s)
                self._resolve(keep=True)
            try:
                item = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                item = None
            if item is not None:
                self._write_frame(item)
            if time.time() - self._last_status > 0.25:
                self._write_status()

    def _poll_cmd_file(self) -> None:
        try:
            with open(self._cmd_path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        stamp = float(d.get("stamp", 0.0))
        if stamp > self._last_cmd_stamp:
            self._last_cmd_stamp = stamp
            self._cmd_q.put(str(d.get("cmd", "")))

    def _apply(self, cmd: str) -> None:
        if cmd == "toggle":
            cmd = {IDLE: "start", RECORDING: "stop", DECIDING: "keep"}.get(self._state, "")
        if cmd == "start":
            self._begin()
        elif cmd == "stop":
            self._end()
        elif cmd == "keep":
            self._resolve(keep=True)
        elif cmd == "discard":
            self._resolve(keep=False)

    # -- state transitions (worker thread only) ----------------------------

    def _begin(self) -> None:
        if self._state != IDLE:
            return
        self._ep_index += 1
        name = f"{datetime.now(_KST).strftime('%Y%m%d-%H%M%S')}_ep{self._ep_index:04d}"
        path = os.path.join(self._pending_dir, name)
        rgb_dir = os.path.join(path, "head_rgb")
        depth_dir = os.path.join(path, "head_depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        states = open(os.path.join(path, "states.jsonl"), "w")
        self._cur = {
            "name": name, "path": path, "rgb_dir": rgb_dir, "depth_dir": depth_dir,
            "states": states, "idx": 0, "t0": time.time(), "final": os.path.join(self.out_dir, name),
        }
        self._write_meta(path, name)
        self._state = RECORDING
        self._recording = True
        logger.info("[record] ● recording {}", name)

    def _end(self) -> None:
        if self._state != RECORDING:
            return
        self._recording = False
        self._state = DECIDING
        self._decide_t0 = time.time()
        logger.info("[record] ■ stopped {} ({} frames) — keep? [y/n]",
                    self._cur["name"], self._cur["idx"])

    def _resolve(self, keep: bool) -> None:
        if self._state not in (RECORDING, DECIDING) or self._cur is None:
            return
        self._recording = False
        # Flush any frames still queued for this take before moving/deleting.
        while True:
            try:
                self._write_frame(self._frame_q.get_nowait())
            except queue.Empty:
                break
        cur = self._cur
        try:
            cur["states"].close()
        except Exception:  # noqa: BLE001
            pass
        if keep:
            preview_frame = self._write_depth_preview(cur)
            self._write_meta(cur["path"], cur["name"], final=True,
                             frames=cur["idx"], duration=time.time() - cur["t0"],
                             depth_preview_frame=preview_frame)
            os.replace(cur["path"], cur["final"])
            self._takes_saved += 1
            self._last_saved = cur["name"]
            logger.success("[record] ✓ kept {} ({} frames) -> {}", cur["name"], cur["idx"], cur["final"])
        else:
            shutil.rmtree(cur["path"], ignore_errors=True)
            logger.warning("[record] ✗ discarded {} ({} frames)", cur["name"], cur["idx"])
        self._cur = None
        self._state = IDLE
        self._write_status()

    # -- writing -----------------------------------------------------------

    def _write_frame(self, item) -> None:
        if self._cur is None:
            return  # stray frame after a take ended — drop it
        rgb, depth, state = item
        cur = self._cur
        i = cur["idx"]
        stem = f"{i:06d}"
        if rgb is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                _atomic_write(os.path.join(cur["rgb_dir"], stem + ".jpg"), buf.tobytes())
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            mm = np.clip(np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0, neginf=0.0), 0, 65535)
            cv2.imwrite(os.path.join(cur["depth_dir"], stem + ".png"), mm.astype(np.uint16))
        row = {
            "i": i,
            "t": state.get("stamp", time.time()),
            "joints": state.get("joints", {}),
            "wrench": state.get("wrench"),
            "suction_cmd": suction_io.is_suction_commanded_on(),
            "gripper_pos": robotiq.commanded_pos(),   # 0=open..255=closed, null=never commanded
        }
        for key in ("ee", "ee_right"):  # suction/configured arm + right gripper
            ee = state.get(key)
            if ee is not None:
                row[key] = {"pos": ee["pos"], "quat_wxyz": _rpy_to_quat_wxyz(ee["rpy"])}
        cur["states"].write(json.dumps(row) + "\n")
        cur["idx"] = i + 1

    def _write_depth_preview(self, cur: dict) -> int | None:
        """Colorize one depth frame (random, near the take's middle) as a sample.

        Reads the already-written raw-mm PNG back from disk and turbo-maps it
        like the dashboard, so the take has one human-viewable depth image
        without bloating every frame. Returns the frame index used, or None.
        """
        n = cur["idx"]
        if n <= 0:
            return None
        lo, hi = n // 3, max(n // 3 + 1, (2 * n) // 3)   # central third
        pick = random.randint(lo, hi - 1) if hi > lo else n // 2
        src = os.path.join(cur["depth_dir"], f"{pick:06d}.png")
        d = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        d = d.astype(np.float32) / 1000.0   # mm -> m
        near, far = _PREVIEW_DEPTH_RANGE_M
        valid = d > 0.0
        norm = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
        norm[~valid] = 0.0
        color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), _DEPTH_CMAP)
        color[~valid] = (0, 0, 0)
        if not cv2.imwrite(os.path.join(cur["path"], "depth_preview.jpg"), color):
            return None
        return pick

    def _write_meta(self, path: str, name: str, final: bool = False,
                    frames: int = 0, duration: float = 0.0,
                    depth_preview_frame: int | None = None) -> None:
        meta = {
            "name": name,
            "created": datetime.now(_KST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "instruction": self.instruction,
            "arm_side": cfg.ARM_SIDE,
            "ee_frame": cfg.EE_FRAME,
            "urdf": cfg.URDF_PATH,
            "action_space": "ee_delta+suction",
            "episode_scope": "manual take (one start->stop = one episode)",
            "ee_rotation": "quat_wxyz, base_link",
            "depth_units": "uint16 millimetres (0 = invalid)",
            "gripper_units": "Robotiq commanded finger pos, 0=open..255=closed, null=never commanded",
            "trace_path": getattr(cfg, "TRACE_PATH", None),
            "note": ("phase labels and commanded-joint actions are recovered offline by "
                     "joining states.jsonl 't' against trace_path rows (t, leg, cmd, actual)."),
        }
        if final:
            meta["frames"] = frames
            meta["duration_s"] = round(duration, 3)
            meta["dropped_frames"] = self._dropped
            if depth_preview_frame is not None:
                meta["depth_preview_frame"] = depth_preview_frame
        _atomic_write(os.path.join(path, "meta.json"), json.dumps(meta, indent=2).encode("utf-8"))

    def _write_status(self) -> None:
        self._last_status = time.time()
        st = {
            "stamp": time.time(),
            "state": self._state,
            "episode": None if self._cur is None else self._cur["name"],
            "frames": 0 if self._cur is None else self._cur["idx"],
            "takes_saved": self._takes_saved,
            "last_saved": self._last_saved,
            "dropped": self._dropped,
            "out_dir": self.out_dir,
        }
        if self._state == RECORDING and self._cur is not None:
            st["elapsed"] = round(time.time() - self._cur["t0"], 1)
        if self._state == DECIDING:
            st["decide_remaining"] = round(max(0.0, self.decide_timeout_s - (time.time() - self._decide_t0)), 1)
        try:
            _atomic_write(self._status_path, json.dumps(st).encode("utf-8"))
        except OSError:
            pass


class KeyListener:
    """Terminal hotkeys for the recorder: SPACE toggle, y keep, n discard.

    Uses cbreak (not raw) so the demo's log output stays readable, and only
    runs on a real TTY. Start it AFTER any input() prompts so it doesn't fight
    the main thread for stdin. Restores the terminal on stop.
    """

    def __init__(self, controller: RecordController) -> None:
        self._c = controller
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._fd: int | None = None
        self._old = None

    def start(self) -> "KeyListener":
        if not sys.stdin.isatty():
            logger.warning("[record] stdin is not a TTY — keyboard control off (use the dashboard buttons).")
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._old = __import__("termios").tcgetattr(self._fd)
        except Exception as e:  # noqa: BLE001
            logger.warning("[record] keyboard control unavailable: {}", e)
            self._old = None
            return self
        self._thread = threading.Thread(target=self._run, name="record-keys", daemon=True)
        self._thread.start()
        logger.info("[record] keys:  SPACE start/stop · y keep · n discard")
        return self

    def _run(self) -> None:
        import select
        import termios
        import tty
        try:
            tty.setcbreak(self._fd)
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == " ":
                    self._c.cmd_toggle()
                elif ch in ("y", "Y"):
                    self._c.cmd_keep()
                elif ch in ("n", "N"):
                    self._c.cmd_discard()
        finally:
            if self._old is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old is not None and self._fd is not None:
            try:
                __import__("termios").tcsetattr(self._fd, __import__("termios").TCSADRAIN, self._old)
            except Exception:  # noqa: BLE001
                pass
