"""Station AprilTags, LOG ONLY (1026): would a tag + a remembered case pose have
put the case where the detection did?

Tags 36h11 lie flat on the table, one right of each bin at the FRONT edge
(left to right: left bin | id0 | source | id1 | target | id2 | task-2 bin | id3)
and one behind each of those, past the bins (id4..id7, cfg.STATION_TAG_PAIRS).

A tag's base pose = its 4 corner rays cut with the fixed table plane
cfg.STATION_TAG_PLANE_Z_M (no PnP: its height swings by cm on small oblique tags).
Reference frames: a PAIR (front + back tag, ~0.59 m apart) = origin at the front
centre, x along front -> back; its yaw held to 0.07 deg between visits (1027),
a single tag's own edges to ~2 deg — 40-50 cm from the case that is 1 vs 15 mm.
A tag whose pair is not complete is its own frame (fallback).
Memory = per (station, frame) one case pose (x, y, yaw) in that frame:

    check()   at a pick / place, with the pose the run actually used: no memory
              yet -> learn it; else log the remembered pose vs the detection,
              for the case centre and for the arm target (pose_key) — pair
              frames first, a single tag only where no pair could answer
    record()  after a CASE place: the case pose rebuilt from the arm's own
              joint readings (inverse of resolve_poses' CASE_PICK) replaces the
              station's memory (all its frames) — the next item there is
              checked against it. The arm still over the bin hides the BACK
              tag then, so tags missing from that view are taken from the last
              check / observe() at the station (the chassis does not move
              between the pre-place look and the release)

Every check / record also logs each visible tag's own base pose, each complete
pair's length and line yaw, and with two pairs in view the angle between them
(all fixed on the table, so any change between visits is measurement error),
and saves the head frame +
joints + those numbers under `out_dir` (one npz per call) for offline replay.

Changes no motion; every call catches its own errors.
"""
from __future__ import annotations

import threading
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from . import config as cfg
from .config import resolve_poses

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _wrap90(a: float) -> float:
    """Case yaw is 180-deg ambiguous (OBB long axis): fold to [-90, 90)."""
    return (a + 90.0) % 180.0 - 90.0


class TagShadow:
    def __init__(self, cg, out_dir: "str | Path | None" = None) -> None:
        self.cg = cg            # case_detection's camera_geometry (live intrinsics)
        self.mem: dict = {}     # (station, frame key) -> (x, y, yaw_rad) in that frame
        self.out_dir = None if out_dir is None else Path(out_dir)
        self._n = 0             # file counter: two calls in one second must not collide
        self._last_seen: dict = {}   # station -> tags of its last check / observe

    @staticmethod
    def frames(seen: dict) -> dict:
        """{key: ((x, y, yaw_rad), member ids)}: every complete cfg pair ("pair a-b",
        front centre + front->back line) and every single tag ("ida")."""
        out = {}
        for a, b in cfg.STATION_TAG_PAIRS:
            if a in seen and b in seen:
                (ax, ay, _), (bx, by, _) = seen[a], seen[b]
                out[f"pair {a}-{b}"] = ((ax, ay, float(np.arctan2(by - ay, bx - ax))),
                                        {a, b})
        for i, t in seen.items():
            out[f"id{i}"] = (t, {i})
        return out

    def export(self, station: str) -> dict:
        """{frame key: [x, y, yaw]} remembered for `station` (handoff file)."""
        return {k: [float(v) for v in pose] for (st, k), pose in self.mem.items()
                if st == station}

    def load(self, frames: dict) -> list:
        """A handoff saved by another task (task 2 placed the case): put each
        remembered frame under the station whose bin holds the case — the pair
        (cfg.STATION_TAG_STATION_PAIRS) with the case STATION_TAG_BIN_Y_M to its
        left (tags lie right of their bin); other pairs that happened to be in
        view are dropped. Returns the stations loaded."""
        lo, hi = cfg.STATION_TAG_BIN_Y_M
        loaded = []
        for station, (a, b) in cfg.STATION_TAG_STATION_PAIRS.items():
            pose = frames.get(f"pair {a}-{b}")
            if pose is None or not lo <= pose[1] <= hi:
                continue
            for k in (f"pair {a}-{b}", f"id{a}", f"id{b}"):
                if k in frames:
                    self.mem[(station, k)] = tuple(frames[k])
            loaded.append(station)
        return loaded

    def knows(self, station: str) -> bool:
        """A case is remembered for `station` in some tag PAIR's frame."""
        return any(st == station and k.startswith("pair") for st, k in self.mem)

    @staticmethod
    def has_pair(seen: dict) -> bool:
        return any(a in seen and b in seen for a, b in cfg.STATION_TAG_PAIRS)

    def predict(self, station: str, seen: dict, allow_single: bool = False):
        """(frame key, remembered case pose (x, y, yaw_rad) in base) from a PAIR
        frame in view, or None. ``allow_single``: else a single tag's frame —
        only good enough for the +-5 cm WINDOW (its ~2 deg yaw is 10-15 mm at the
        case), never for a pick / place aim."""
        fr = self.frames(seen)
        for single in ((False, True) if allow_single else (False,)):
            for k, (tag, _) in sorted(fr.items()):
                if k.startswith("pair") != single and (station, k) in self.mem:
                    return k, self._to_base(tag, *self.mem[(station, k)])
        return None

    def _log_tags(self, station: str, label: str, seen: dict, fr: dict) -> None:
        """Each tag's base pose, each pair's length + line yaw, the angle between pairs."""
        logger.info("[tagshadow] {} {} tags: {}", station, label, ", ".join(
            "id{} ({:.4f},{:+.4f}) yaw {:+.2f}".format(i, x, y, np.degrees(a))
            for i, (x, y, a) in sorted(seen.items())))
        pairs = [f"pair {a}-{b}" for a, b in cfg.STATION_TAG_PAIRS if a in seen and b in seen]
        for a, b in cfg.STATION_TAG_PAIRS:
            if a in seen and b in seen:
                logger.info("[tagshadow] {} {} pair {}-{}: {:.1f} mm long, line yaw "
                            "{:+.3f}deg", station, label, a, b,
                            float(np.hypot(seen[b][0] - seen[a][0],
                                           seen[b][1] - seen[a][1])) * 1000.0,
                            np.degrees(fr[f"pair {a}-{b}"][0][2]))
        for k1, k2 in combinations(pairs, 2):
            logger.info("[tagshadow] {} {} {} vs {}: {:+.3f}deg", station, label, k1, k2,
                        float(np.degrees(np.angle(np.exp(1j * (fr[k2][0][2]
                                                               - fr[k1][0][2]))))))

    def _save(self, kind: str, rgb, q_torso, q_head, station: str, label: str,
              center, seen: dict) -> None:
        """One npz per check / record (file write on a thread)."""
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ids = sorted(seen)
        d = dict(head_rgb=rgb, q_torso=np.asarray(q_torso, dtype=float),
                 q_head=np.asarray(q_head, dtype=float), kind=kind, station=station,
                 label=label, center=np.asarray(center, dtype=float),
                 tag_ids=np.asarray(ids, dtype=int),
                 tag_poses=np.asarray([seen[i] for i in ids], dtype=float).reshape(-1, 3),
                 plane_z=float(cfg.STATION_TAG_PLANE_Z_M), t=time.time())
        self._n += 1
        f = self.out_dir / f"{self._n:03d}_{time.strftime('%H%M%S')}_{station}_{label}_{kind}.npz"
        threading.Thread(target=np.savez_compressed, args=(f,), kwargs=d,
                         daemon=True).start()

    def tags(self, rgb, q_torso, q_head) -> dict:
        """{id: (x, y, yaw_rad)} base pose of every tag in the head frame.
        Found at 1x AND on a 2x upscale (the back tags image at ~25-30 px and
        drop in and out at 1x: 49 saved 1027 frames, 82 tags at 1x / 101 at
        2x, +18 ms); corners refined on the 1x image either way."""
        cg = self.cg
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        det = cv2.aruco.ArucoDetector(_DICT, cv2.aruco.DetectorParameters())
        found: dict = {}
        for scale in (2.0, 1.0):            # 1x last: it wins where both found a tag
            g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale,
                                                     interpolation=cv2.INTER_CUBIC)
            cs_, ids_, _ = det.detectMarkers(g)
            for c, i in zip(cs_, [] if ids_ is None else ids_.ravel()):
                found[int(i)] = c / scale
        if not found:
            return {}
        corners, ids = list(found.values()), np.array(list(found.keys()))
        K = np.array([[cg.ZED_FX, 0, cg.ZED_CX], [0, cg.ZED_FY, cg.ZED_CY], [0, 0, 1]])
        T = cg.zed_left_camera_pose_from_joints(q_torso, q_head)
        z = float(cfg.STATION_TAG_PLANE_Z_M)
        out = {}
        for c, i in zip(corners, ids.ravel()):
            uv = c[0].astype(np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(gray, uv, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
            uv = uv.reshape(4, 2).astype(np.float64)
            d = T[:3, :3] @ (np.linalg.inv(K) @ np.c_[uv, np.ones(4)].T)
            P = T[:3, 3:4] + d * ((z - T[2, 3]) / d[2])        # corners on the table
            ex = (P[:2, 1] - P[:2, 0]) + (P[:2, 2] - P[:2, 3])  # tag +x edge
            out[int(i)] = (float(P[0].mean()), float(P[1].mean()),
                           float(np.arctan2(ex[1], ex[0])))
        return out

    @staticmethod
    def _to_tag(tag, x, y, yaw):
        tx, ty, ta = tag
        c, s = np.cos(ta), np.sin(ta)
        return (c * (x - tx) + s * (y - ty), -s * (x - tx) + c * (y - ty), yaw - ta)

    @staticmethod
    def _to_base(tag, lx, ly, lyaw):
        tx, ty, ta = tag
        c, s = np.cos(ta), np.sin(ta)
        return (tx + c * lx - s * ly, ty + s * lx + c * ly, lyaw + ta)

    def check(self, rgb, q_torso, q_head, station: str, label: str, center,
              pose_key: "str | None" = None) -> None:
        """`center` = the (x, y, z, yaw) case centre the run is using right now."""
        # COPY first: the camera reuses this buffer — 1027 frames saved after the
        # detection held a LATER image (the marginal back tag found / not found)
        rgb = np.array(rgb, copy=True)
        try:
            seen = self.tags(rgb, q_torso, q_head)
            self._last_seen[station] = seen
            self._save("check", rgb, q_torso, q_head, station, label, center, seen)
            if not seen:
                logger.warning("[tagshadow] {} {}: no tag in view", station, label)
                return
            fr = self.frames(seen)
            self._log_tags(station, label, seen, fr)
            # pairs first; a single tag only where no pair with a memory covered it
            answered: set = set()
            for k in sorted(fr, key=lambda k: not k.startswith("pair")):
                tag, members = fr[k]
                if (station, k) not in self.mem or members <= answered:
                    continue
                answered |= members
                px, py, pyaw = self._to_base(tag, *self.mem[(station, k)])
                # onto the detection's 180-deg branch (else the grab side flips)
                pyaw = center[3] + np.radians(_wrap90(np.degrees(pyaw - center[3])))
                msg = ("[tagshadow] {} {} {}: tag says ({:.4f},{:+.4f}) yaw {:+.1f}, "
                       "detection ({:.4f},{:+.4f}) yaw {:+.1f} -> case dx {:+.1f} dy {:+.1f} "
                       "mm dyaw {:+.2f}deg")
                args = [station, label, k, px, py, np.degrees(pyaw), center[0], center[1],
                        np.degrees(center[3]), (px - center[0]) * 1000,
                        (py - center[1]) * 1000,
                        _wrap90(np.degrees(pyaw - center[3]))]
                if pose_key is not None:
                    a = resolve_poses((px, py, center[2], pyaw))[pose_key]
                    b = resolve_poses(tuple(center))[pose_key]
                    msg += ", {} target dx {:+.1f} dy {:+.1f} mm"
                    args += [pose_key, (a[0] - b[0]) * 1000, (a[1] - b[1]) * 1000]
                logger.info(msg, *args)
            if not answered:
                logger.info("[tagshadow] {} {}: nothing remembered for the frames in view "
                            "({})", station, label, ", ".join(sorted(fr)))
            for k, (tag, _) in sorted(fr.items()):
                if (station, k) not in self.mem:
                    self.mem[(station, k)] = self._to_tag(tag, center[0], center[1],
                                                          center[3])
                    logger.info("[tagshadow] {} {} {}: learned the case from the "
                                "detection ({:.4f},{:+.4f}) yaw {:+.1f}deg", station,
                                label, k, center[0], center[1], np.degrees(center[3]))
        except Exception as e:  # noqa: BLE001 — log only, never stops a run
            logger.warning("[tagshadow] {} {} check failed ({})", station, label, e)

    def observe(self, rgb, q_torso, q_head, station: str, label: str) -> None:
        """Tags only, no comparison (a place with nothing remembered to check
        against, e.g. the first case into the empty left bin) — for record()."""
        rgb = np.array(rgb, copy=True)       # see check()
        try:
            seen = self.tags(rgb, q_torso, q_head)
            self._last_seen[station] = seen
            self._save("observe", rgb, q_torso, q_head, station, label, (0, 0, 0, 0), seen)
            self._log_tags(station, label, seen, self.frames(seen))
        except Exception as e:  # noqa: BLE001
            logger.warning("[tagshadow] {} {} observe failed ({})", station, label, e)

    def record(self, rgb, q_torso, q_head, station: str, label: str,
               ee_pos, ee_rpy, aimed) -> None:
        """After a CASE place: case pose from the EE (joint readings), into memory.
        `aimed` = the (x, y, z, yaw) centre the place was aimed at, for the log."""
        rgb = np.array(rgb, copy=True)       # see check()
        try:
            yaw = float(ee_rpy[2]) - cfg.GRASP_YAW
            c, s = np.cos(yaw), np.sin(yaw)
            gx, gy = cfg.CASE_GRAB_OFFSET[0], cfg.CASE_GRAB_OFFSET[1]
            x = float(ee_pos[0]) - (c * gx - s * gy)
            y = float(ee_pos[1]) - (s * gx + c * gy)
            logger.info("[tagshadow] {} {}: case from the arm at release ({:.4f},{:+.4f}) "
                        "yaw {:+.1f} vs aimed ({:.4f},{:+.4f}) yaw {:+.1f} -> dx {:+.1f} "
                        "dy {:+.1f} mm dyaw {:+.2f}deg", station, label, x, y,
                        _wrap90(np.degrees(yaw)), aimed[0], aimed[1],
                        _wrap90(np.degrees(aimed[3])), (x - aimed[0]) * 1000,
                        (y - aimed[1]) * 1000, _wrap90(np.degrees(yaw - aimed[3])))
            now = self.tags(rgb, q_torso, q_head)
            self._save("record", rgb, q_torso, q_head, station, label, (x, y, 0.0, yaw), now)
            seen = {**self._last_seen.get(station, {}), **now}
            if not seen:
                logger.warning("[tagshadow] {} {}: no tag in view — memory unchanged",
                               station, label)
                return
            if len(seen) > len(now):
                logger.info("[tagshadow] {} {}: id{} hidden now, from the pre-place view",
                            station, label, ",".join(str(i) for i in sorted(set(seen) - set(now))))
            fr = self.frames(seen)
            self._log_tags(station, label, seen, fr)
            # a new case on top: every older frame's memory here is the old case
            for key in [key for key in self.mem if key[0] == station]:
                del self.mem[key]
            for k, (tag, _) in sorted(fr.items()):
                self.mem[(station, k)] = self._to_tag(tag, x, y, yaw)
            logger.info("[tagshadow] {} {}: memory <- the placed case ({})", station,
                        label, ", ".join(sorted(fr)))
        except Exception as e:  # noqa: BLE001
            logger.warning("[tagshadow] {} {} record failed ({})", station, label, e)
