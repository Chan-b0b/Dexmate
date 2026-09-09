"""Live BEV viewer for the trained BEV detectors — one active at a time.

Every OBB model under runs/detect/ (bin / box / case / cylinder / showbox) is
loaded up front, but only the SELECTED one runs each frame; 1-5 or Tab switches
which. That is deliberate: the BEV canvas can only be warped at ONE plane, and
the targets do not share a height (the bin's inner bottom is not the paper box's
rim, and the show bin's rim is 25 cm above either), so a canvas that suits them
all suits none of them well and the numbers for five models at once are
unreadable anyway.

With a single active target the warp plane can be taken from the ZED DEPTH and
LOCKED ONTO THAT TARGET, so nothing has to be measured or passed in:

  1. SEED — with no history, deproject the depth map to base_link, keep the
     points inside the BEV ROI and take the histogram MODE of their z: the
     dominant flat surface in the canvas. (The median would sit between two
     surfaces on a bimodal scene; the mode picks the larger one.)
  2. DETECT — warp at that plane, run the active model, keep every box.
  3. MEASURE + LOCK — depth_plane.plane_from_depth measures the surface under
     the detected center and bev.reproject_plane moves the center onto that
     height (an exact homothety about the camera centre, see bev.py — no
     re-warp, no re-detection); the metric size is rescaled by the same factor
     and yaw is left alone, being plane-invariant. That measurement becomes the
     next frame's warp plane for this model, so the canvas converges onto the
     active target's own surface within a frame or two and each model keeps its
     own remembered height across switches.

The header shows the active model, the plane, and where the plane came from.
Each box shows its own measured z and how far the center moved because of it: a
large "d" means the canvas plane alone would have put that object tens of mm
off, and only the reprojected center is right.

A red X marks the EE TARGET — where ik_demo would actually send the arm for
that detection, which is never the detected center (see ee_target: the cup
grabs a case off-center, the bin place carries a measured bias, the box pick
pinches a wall). The offsets are read live from LGES.ik_demo.config, so the
marker tracks whatever the runs are tuned to.

With nothing detected the RAW camera frame is shown instead of the canvas: an
empty BEV says nothing, and if the plane is off it says something wrong, while
the raw frame at least shows whether the target is in view at all.

Robot-side (needs the head camera with depth enabled).
Keys: 1-5 or Tab switch model, r forget the locked plane, s snap, q/Esc quit.

    python live_detect_bev.py                  # starts on case
    python live_detect_bev.py --model bin --conf 0.5
    python live_detect_bev.py --model box --plane 0.699   # pin, no depth seeding

--serve streams the same annotated canvas over HTTP (MJPEG) with the same keys
in the browser, for shells with no display. The installed cv2 is the HEADLESS
build (opencv-python-headless overwrote opencv-python's cv2 module), so
imshow/namedWindow raise "The function is not implemented" no matter what
DISPLAY says — on this robot --serve is the working path:

    python live_detect_bev.py --serve 8088     # then open http://<robot>:8088
"""

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "perception"))

import config as cfg
import bev
import camera_geometry as cg
import depth_plane as dp
from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot
from utils import set_head_pitch

# ik_demo's config, for the EE target each detection would actually be sent to.
# Read LIVE rather than copied: these are tuning values that move, and a viewer
# showing a stale offset is worse than one showing none. Its package name is
# LGES.ik_demo.config, so it does not collide with case_detection's own
# top-level `config` (imported above as cfg). Optional — the viewer still runs
# without it, just with no EE markers.
try:
    sys.path.insert(0, str(_HERE.parents[1]))
    from LGES.ik_demo import config as ikcfg
except Exception as _e:  # noqa: BLE001 - a viewer must not die on this
    ikcfg, _IK_ERR = None, str(_e)

# The trained detectors, with the colour each draws in (BGR). Order is the
# 1..N key order, so existing keys keep their meaning when one is appended.
# Note the paths are not uniform (case has weight/ not weights/, and the two
# 0906 additions sit directly in their run dir) — they are what train.py wrote.
MODELS: dict[str, tuple[str, tuple[int, int, int]]] = {
    "bin": ("runs/detect/bin/weights/bin_detector.pt", (255, 160, 0)),
    "box": ("runs/detect/box/weights/box_detector.pt", (0, 165, 255)),
    "case": ("runs/detect/case/weight/case_detector.pt", (0, 255, 0)),
    "cylinder": ("runs/detect/cylinder/cylinder_detector.pt", (255, 0, 255)),
    "showbox": ("runs/detect/showbox/showbox_detector.pt", (0, 255, 255)),
}


def load_models() -> dict:
    """Load every detector once, up front. Switching model at runtime must not
    stall the stream on a cold load, and they are small.

    All of them are YOLO-OBB trained on the same metric BEV canvas, despite
    living under runs/detect/ rather than runs/obb/."""
    from ultralytics import YOLO  # noqa: PLC0415 (optional heavy dep)
    out = {}
    for name, (rel, _) in MODELS.items():
        path = _HERE / rel
        if not path.exists():
            raise FileNotFoundError(f"{name} weights not found at {path}")
        out[name] = YOLO(str(path))
    return out


def _get_frame(robot):
    """One time-synced (rgb, depth) pair from the head ZED — one get_obs call so
    the depth the plane comes from is the depth of the frame we detect on."""
    obs = robot.sensors.head_camera.get_obs(obs_keys=["left_rgb", "depth"])
    rgb, depth = obs.get("left_rgb"), obs.get("depth")
    rgb = rgb.get("data") if isinstance(rgb, dict) else rgb
    depth = depth.get("data") if isinstance(depth, dict) else depth
    return rgb, depth


def _joints(robot):
    return (np.asarray(robot.torso.get_joint_pos(), dtype=np.float64),
            np.asarray(robot.head.get_joint_pos(), dtype=np.float64))


def seed_plane(depth: np.ndarray, rgb_shape, q_torso, q_head,
               stride: int = 8, bin_m: float = 0.02):
    """Base z of the dominant flat surface inside the BEV ROI, from depth.

    Returns (z, n_points_in_roi), or (None, n) when too little valid depth
    landed in the ROI. Depth is registered to the left image but need not be
    published at the same size, so pixel indices are scaled into the intrinsics
    frame first (same correction as depth_plane.sample_depth)."""
    dh, dw = depth.shape[:2]
    rh, rw = rgb_shape[0], rgb_shape[1]
    vs, us = np.mgrid[0:dh:stride, 0:dw:stride]
    d = depth[vs, us].astype(np.float64)
    ok = np.isfinite(d) & (d > cfg.DEPTH_MIN_M) & (d < cfg.DEPTH_MAX_M)
    if int(ok.sum()) < 50:
        return None, 0
    u = us[ok] * (rw / float(dw))
    v = vs[ok] * (rh / float(dh))
    d = d[ok]
    K = bev.intrinsic_matrix()
    p_cam = np.stack([(u - K[0, 2]) / K[0, 0] * d,
                      (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)])
    P = (cg.zed_left_camera_pose_from_joints(q_torso, q_head) @ p_cam)[:3].T
    x0, x1 = cfg.BEV_X_RANGE
    y0, y1 = cfg.BEV_Y_RANGE
    z = P[(P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1), 2]
    if z.size < 50:
        return None, int(z.size)
    edges = np.arange(z.min(), z.max() + bin_m, bin_m)
    if edges.size < 3:
        return float(np.median(z)), int(z.size)
    hist, _ = np.histogram(z, bins=edges)
    k = int(np.argmax(hist))                      # the fullest height bin
    return float(np.median(z[(z >= edges[k]) & (z < edges[k + 1])])), int(z.size)


def clamp_plane(z: float, q_torso, q_head) -> float:
    """Keep the warp plane a sane distance below the camera. The homothety
    scale is 1/(z - C_z), so a plane creeping up to the camera height blows the
    canvas up without bound; the floor is a loose sanity bound on the other
    side. Only ever trips on a bad depth measurement."""
    ceiling = float(bev.camera_centre(q_torso, q_head)[2]) - 0.20
    return float(np.clip(z, 0.05, ceiling))


def _rotated(X: float, Y: float, yaw_deg: float, off) -> tuple[float, float]:
    """Object-local (dx, dy) offset rotated by the object's yaw into base_link —
    the same transform ik_demo's resolve_poses.src applies."""
    yaw = np.deg2rad(float(yaw_deg))
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return (X + c * off[0] - s * off[1], Y + s * off[0] + c * off[1])


def ee_target(cls: str, X: float, Y: float, yaw_deg: float, dims_m,
              warped_xy=None, plane_z: float | None = None,
              cam=None) -> tuple[tuple[float, float], str] | None:
    """base_link (x, y) the arm is actually sent to for this detection, with a
    short label — or None if this class has no EE target in ik_demo.

    The detected CENTER is not where the EE goes: every flow applies its own
    offset, so a viewer that marks only the center hides the number that
    decides reach. Per ik_demo's config:

      case             cup grab point, CASE_GRAB_OFFSET rotated into the case
                       frame (config/geometry.py, resolve_poses.src) — but only
                       AFTER the 180-deg wrap below
      bin              seed place point, SEED_BIN_CENTER_OFFSET — a BASE-frame
                       measured bias, NOT rotated (case_to_bin._bin_place_center)
      bin_top          BIN lid cup grab point, LID_GRAB_OFFSET_M rotated into
                       the lid's frame (config/lid.py, chassis_sequence lid pick)
      box_top          PAPER lid cup grab point, BOX_LID_GRAB_OFFSET_M — the
                       CENTRE (0,0), so the marker sits on the detected centre.
                       These two were ONE branch until 0906 and box_top drew the
                       bin lid's -50,+20mm, 54mm off the real aim
      box              the box is wider than the Robotiq opening, so the pick
                       pinches the LONG WALL on the robot's right (base -y):
                       half the short side out from the center, less
                       BOX_GRASP_EDGE_INSET_M (box_pick.box_pose_from_detection)
      cylinder         gripper pinch point: the centre reprojected to the
                       cylinder's MID height + STAND_CYL_DET_OFFSET_XY
      show             place point: the centre reprojected to the bin RIM
                       + STAND_BIN_DET_OFFSET_XY

    cylinder / show DO NOT USE DEPTH, and that is not an oversight in this
    viewer — show_detect.detect_scene reprojects both to a FIXED MODEL height
    (config/show.py) and this mirrors it, because for those two a measured
    surface is the wrong quantity:
      * a STANDING cylinder smears across the BEV from its base to its top, so
        its OBB centre sits at about mid-height. Depth under that centre reads
        either the cylinder's top or the desk beside it, and neither is the
        mid-height the flow aims at (STAND_DESK_Z_M + height/2 = 0.751).
      * the show bin's OBB is its RIM, so its centre already lives at rim
        height (STAND_BIN_RIM_Z_M = 0.866), not on any surface depth can see
        under it.
    Both are reprojected from the AS-WARPED centre, so the result is
    independent of whatever plane this viewer happens to be warping at
    (reproject_plane is exact between any two planes for the same camera ray) —
    which is what lets the depth lock stay on for the canvas without moving
    these two markers.

    The classes handle the OBB's 180-deg long-axis ambiguity DIFFERENTLY,
    and each one here follows its own flow rather than a shared convention:
      * case  WRAPS to [-90, 90) first (_center_from_det), because a top-down
        suction grasp is symmetric under a 180-deg case flip. Without the wrap
        the marker lands on the OPPOSITE side of the case (2 * |offset|,
        ~117 mm) for every detection above 90 deg.
      * lids WRAP via ikcfg.canonical_lid_yaw, the single definition the run
        itself folds every detection with (chassis_sequence._detect_lid_xy).
        Until 0906 this branch deliberately did NOT wrap, to stay faithful to a
        _lid_pick_aim that rotated by the raw [0,180) yaw — but that was the bug,
        not the convention: the same physical lid read 175.9 deg one day and 1.5
        the next, mirroring the cup point about the lid centre (a 100mm swing in
        x at the taught -50,+20mm offset). Now that the run folds, a viewer that
        did not would draw the mirror image of where the arm is going.
      * box is ambiguity-free: box_pose_from_detection re-points its across-box
        axis at the robot's right, so the wall it picks never depends on which
        end the detector called the long axis.
      * bin does not rotate at all, so its marker is yaw-invariant.
      * cylinder / show do not rotate either (both offsets are base-frame), so
        the ambiguity cannot reach them.
    """
    if ikcfg is None:
        return None
    if cls in ("cylinder", "show"):
        if warped_xy is None or plane_z is None or cam is None:
            return None            # cannot reproject without the warp context
        if cls == "cylinder":
            z = (float(ikcfg.STAND_DESK_Z_M)
                 + 0.5 * float(ikcfg.STAND_OBJECTS["cylinder"]["height"]))
            off, what = ikcfg.STAND_CYL_DET_OFFSET_XY, "pinch"
        else:
            z, off, what = ikcfg.STAND_BIN_RIM_Z_M, ikcfg.STAND_BIN_DET_OFFSET_XY, "place"
        xy = bev.reproject_plane(warped_xy, float(plane_z), float(z), cam)
        return (float(xy[0]) + float(off[0]), float(xy[1]) + float(off[1])), what
    if cls == "case":
        wrapped = yaw_deg - 180.0 if yaw_deg >= 90.0 else yaw_deg
        return _rotated(X, Y, wrapped, ikcfg.CASE_GRAB_OFFSET[:2]), "cup"
    if cls == "bin":
        ox, oy = ikcfg.SEED_BIN_CENTER_OFFSET
        return (X + float(ox), Y + float(oy)), "place"
    if cls in ("bin_top", "box_top"):
        off = (ikcfg.LID_GRAB_OFFSET_M if cls == "bin_top"
               else ikcfg.BOX_LID_GRAB_OFFSET_M)
        return _rotated(X, Y, ikcfg.canonical_lid_yaw(yaw_deg), off), "cup"
    if cls == "box":
        yaw = np.deg2rad(float(yaw_deg))
        u = np.array([-np.sin(yaw), np.cos(yaw)])   # across the box, unit
        if u[1] > 0.0:                              # point to the robot's right
            u = -u
        half = float(dims_m[1]) / 2.0 - float(ikcfg.BOX_GRASP_EDGE_INSET_M)
        return (X + float(u[0]) * half, Y + float(u[1]) * half), "grip"
    return None


def detect(rgb, depth, q_torso, q_head, model, conf: float,
           plane_z: float, color) -> tuple[np.ndarray, list[dict]]:
    """Warp at ``plane_z``, run ONE model, refine every detection with depth.

    Detections carry both centers: ``base_xy`` AS WARPED (what the canvas says)
    and ``base_xy_z`` after reprojection onto the surface height measured under
    them. dims_m is rescaled by the same homothety factor; yaw is unchanged
    because angles are plane-invariant. A detection whose depth window came up
    empty keeps its as-warped center and reports z_face None rather than
    inventing a height."""
    mapper = bev.build_mapper(q_torso, q_head, plane_z)
    bev_img = mapper.warp(rgb)
    bev_bgr = cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR)
    res = model.predict(bev_bgr, conf=conf, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return bev_bgr, []

    C = bev.camera_centre(q_torso, q_head)
    s = 1.0 / cfg.BEV_PX_PER_M
    xywhr = res.obb.xywhr.cpu().numpy()
    polys = res.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    confs = res.obb.conf.cpu().numpy()
    clss = res.obb.cls.cpu().numpy().astype(int)

    dets: list[dict] = []
    for i in range(len(confs)):
        cx, cy, w, h, r = xywhr[i]
        yaw = float(np.rad2deg(float(r)) + (90.0 if w < h else 0.0)) % 180.0
        X, Y = mapper.bev_px_to_base(float(cx), float(cy))
        long_m, short_m = float(max(w, h)) * s, float(min(w, h)) * s

        # The canvas plane is not necessarily this object's plane — measure it.
        z_face, n_px, spread = dp.plane_from_depth(
            depth, rgb.shape, q_torso, q_head, (X, Y), plane_z) \
            if depth is not None else (None, 0, 0.0)
        if z_face is None:
            xy, k = (X, Y), 1.0
        else:
            xy = bev.reproject_plane((X, Y), plane_z, z_face, C)
            k = (z_face - C[2]) / (plane_z - C[2])
        name = model.names[clss[i]]
        yaw_base = mapper.bev_yaw_to_base(yaw)
        yaw_raw = yaw_base
        if name in ("bin_top", "box_top") and ikcfg is not None:
            # Show and use what the RUN reads. chassis_sequence._detect_lid_xy
            # folds every lid detection to the near-0 branch before anything
            # downstream sees it, so a viewer carrying the raw [0,180) OBB value
            # prints a different angle than the run logs for the SAME lid — and
            # on box_top, whose grab offset is (0,0), the fold changes nothing
            # else, so the displayed angle was the only place it could show.
            yaw_base = float(ikcfg.canonical_lid_yaw(yaw_base))
        dims = (long_m * k, short_m * k)
        # EE target from the REFINED center: that is the pose ik_demo would act
        # on, so the marker has to move with the depth correction too.
        dets.append(dict(
            cls=name, conf=float(confs[i]), poly=polys[i],
            center_px=(float(cx), float(cy)), base_xy=(X, Y),
            base_xy_z=(float(xy[0]), float(xy[1])),
            yaw=yaw_base, yaw_raw=yaw_raw,
            z_face=z_face, dims_m=dims, n_px=n_px, spread=spread,
            color=color,
            ee=ee_target(name, float(xy[0]), float(xy[1]), yaw_base, dims,
                         warped_xy=(X, Y), plane_z=plane_z, cam=C),
        ))
    dets.sort(key=lambda d: -d["conf"])
    return bev_bgr, dets


_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _folded(d) -> "float | None":
    """The raw OBB yaw when detect() folded it (lid classes), else None — so
    both readouts can say "this angle is not what the detector printed"."""
    raw = d.get("yaw_raw")
    if raw is None or abs(float(d["yaw"]) - float(raw)) < 1e-6:
        return None
    return float(raw)


def _text(img, s: str, org, col, scale: float = 0.45) -> None:
    """Text on a darkened plate — the BEV is a photo, so plain text lands on
    whatever colour the scene happens to be (bright orange bins included).

    A plate rather than an outline pass: putText's glyph advance depends on
    ``thickness`` (366 px vs 385 px for one legend line at 0.5), so drawing the
    same string at thickness 3 then 1 does not overprint, it renders doubled and
    progressively offset."""
    (tw, th), base = cv2.getTextSize(s, _FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    x0, y0 = max(0, x - 2), max(0, y - th - 2)
    x1, y1 = min(img.shape[1], x + tw + 2), min(img.shape[0], y + base + 1)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * 0.35).astype(img.dtype)
    cv2.putText(img, s, (x, y), _FONT, scale, col, 1, cv2.LINE_AA)


def _flip_v(v, h):
    """Canvas row -> DISPLAY row. bev.py lays the canvas out with v increasing
    along base +y, so the robot's LEFT sits at the BOTTOM while forward runs to
    the RIGHT — that maps z-up INTO the screen, i.e. a top-down view seen from
    UNDER the floor, and left/right read backwards on it. 0906: an operator
    read a cup offset aimed 20mm to the robot's right off the picture as being
    to the left, because on the un-flipped canvas it is drawn upward.

    draw() flips the image once and every coordinate goes through here, so the
    view becomes a real top-down: forward -> RIGHT, robot LEFT -> UP,
    robot RIGHT -> DOWN. Detection is untouched — the models still see the
    canvas exactly as bev.warp() produced it."""
    return h - 1 - v


def _grid(disp: np.ndarray, S: float) -> None:
    """10 cm base-frame grid on the BEV image (dark green), as live_bev."""
    x0, x1 = cfg.BEV_X_RANGE
    y0, y1 = cfg.BEV_Y_RANGE
    s = cfg.BEV_PX_PER_M * S
    h = disp.shape[0]
    for X in np.arange(np.ceil(x0 * 10) / 10, x1, 0.1):
        u = int((X - x0) * s)
        cv2.line(disp, (u, 0), (u, h), (0, 90, 0), 1)
    for Y in np.arange(np.ceil(y0 * 10) / 10, y1, 0.1):
        v = int(_flip_v((Y - y0) * s, h))
        cv2.line(disp, (0, v), (disp.shape[1], v), (0, 90, 0), 1)


def _base_to_px(X: float, Y: float, S: float = 1.0,
                h: "int | None" = None) -> tuple[int, int]:
    """base_link (x, y) -> DISPLAY pixel, vertical flip included (_flip_v).
    ``h`` is the display height; derived from the ROI when not given, which is
    the same arithmetic bev.py sizes the canvas with."""
    x0, _ = cfg.BEV_X_RANGE
    y0, y1 = cfg.BEV_Y_RANGE
    s = cfg.BEV_PX_PER_M * S
    if h is None:
        h = int(round((y1 - y0) * s))
    return (int(round((X - x0) * s)), int(round(_flip_v((Y - y0) * s, h))))


_EE_COLOR = (60, 60, 255)      # red X — one fixed colour, never a model's


def draw(bev_bgr, dets, active: str, plane_z: float, src: str, fps: float,
         grab_ms: float, det_ms: float, scale: float = 1.0,
         raw_bgr=None) -> np.ndarray:
    """OBBs + EE targets on the canvas, one stacked legend row per detection.

    With NOTHING detected there is nothing to read off the warped canvas, and a
    BEV of the wrong plane is actively misleading — so ``raw_bgr`` (the camera
    frame as it came in) is shown instead, which is what tells you whether the
    target is even in view.

    The numbers go in the legend, not next to the boxes: two boxes of the same
    model can overlap (a case sitting in a bin, two stacked lids) and their
    labels would land in the same few pixels and render on top of each other.
    Only a short index tag rides on the box itself.

    Everything is drawn at ``scale``: the image is resized first and every
    coordinate, marker, line and font scaled with it, so the overlay stays
    crisp instead of being an upscaled blur of a small render."""
    S = float(scale)
    empty = not dets and raw_bgr is not None
    base = raw_bgr if empty else bev_bgr
    disp = (cv2.resize(base, None, fx=S, fy=S, interpolation=cv2.INTER_LINEAR)
            if S != 1.0 else base.copy())
    th = max(1, int(round(2 * S)))                  # line thickness
    fs, fh = 0.5 * S, int(round(16 * S))            # font scale, legend pitch
    if not empty:
        # The BEV canvas only — the raw camera fallback is a real photo and
        # must stay the way the lens saw it. See _flip_v for why.
        disp = cv2.flip(disp, 0)
        _grid(disp, S)
    hd = disp.shape[0]
    col = MODELS[active][1]
    tabs = "  ".join(f"[{i + 1}]{n}" + ("*" if n == active else "")
                     for i, n in enumerate(MODELS))
    _text(disp, f"{tabs}   z={plane_z:.4f} ({src})   {len(dets)} det"
                f"{'' if empty else '   [+x fwd ->right,  +y left ->UP]'}",
          (8, int(18 * S)), col, fs)
    if empty:
        _text(disp, "no detection — showing raw left_rgb "
                    f"({raw_bgr.shape[1]}x{raw_bgr.shape[0]})",
              (8, int(18 * S) + fh), (255, 255, 255), 0.44 * S)
        _text(disp, f"{fps:.1f} fps  grab {grab_ms:.0f}ms  det {det_ms:.0f}ms   "
              f"1-{len(MODELS)}|Tab model  r replane  s snap  q quit",
              (8, disp.shape[0] - int(8 * S)), (255, 255, 255), 0.45 * S)
        return disp

    for k, d in enumerate(dets):
        poly = (d["poly"] * S).astype(np.float64)
        poly[:, 1] = _flip_v(poly[:, 1], hd)
        cv2.polylines(disp, [poly.astype(np.int32)], True, col, th)
        cu = int(d["center_px"][0] * S)
        cvv = int(_flip_v(d["center_px"][1] * S, hd))
        cv2.circle(disp, (cu, cvv), max(2, int(3 * S)), col, -1)   # as warped
        X, Y = d["base_xy_z"]
        ru, rvv = _base_to_px(X, Y, S, hd)
        # Cross = the center after depth reprojection; the line back to the dot
        # is the bias the canvas plane alone would have cost.
        cv2.drawMarker(disp, (ru, rvv), col, cv2.MARKER_CROSS, int(18 * S), th)
        if abs(ru - cu) + abs(rvv - cvv) > 3:
            cv2.line(disp, (cu, cvv), (ru, rvv), col, 1)

        # The EE target: an X where the arm would actually be sent, joined to
        # the center it was derived from.
        ee_s = ""
        if d["ee"] is not None:
            (ex, ey), what = d["ee"]
            eu, evv = _base_to_px(ex, ey, S, hd)
            cv2.line(disp, (ru, rvv), (eu, evv), _EE_COLOR, 1)
            cv2.drawMarker(disp, (eu, evv), _EE_COLOR,
                           cv2.MARKER_TILTED_CROSS, int(22 * S), th)
            _text(disp, what, (eu + int(12 * S), evv + int(5 * S)),
                  _EE_COLOR, 0.42 * S)
            ee_s = f"  EE[{what}] ({ex:.3f},{ey:+.3f})"

        tag = f"{k}:{d['cls']}"
        _text(disp, tag,
              (int(np.clip(poly[:, 0].min(), 2, disp.shape[1] - 90 * S)),
               int(np.clip(poly[:, 1].min() - 5 * S, 12 * S, hd - 4))), col, fs)
        off = np.hypot(X - d["base_xy"][0], Y - d["base_xy"][1]) * 1000
        zs = "z ?" if d["z_face"] is None else f"z{d['z_face']:.3f} d{off:.0f}mm"
        yw = f"{d['yaw']:5.1f}deg"
        if _folded(d) is not None:
            yw += f"(obb{d['yaw_raw']:.0f})"
        _text(disp, f"{tag} {d['conf']:.2f} ({X:.3f},{Y:+.3f}) {yw} "
              f"{d['dims_m'][0]:.2f}x{d['dims_m'][1]:.2f}m {zs}{ee_s}",
              (8, int(38 * S) + fh * k), col, 0.44 * S)

    _text(disp, f"{fps:.1f} fps  grab {grab_ms:.0f}ms  det {det_ms:.0f}ms   "
          f"1-{len(MODELS)}|Tab model  r replane  s snap  q quit",
          (8, disp.shape[0] - int(8 * S)), (255, 255, 255), 0.45 * S)
    return disp


# ---------------------------------------------------------------------------
# Headless output: the annotated canvas as an MJPEG stream, keys over HTTP.
#
# The detection loop stays in the main thread (it owns the robot handle and the
# models); the server runs in a daemon thread and only ever touches _Control,
# so a browser connecting, stalling or dropping cannot affect the loop rate.
# Browser keys are pushed back as the SAME key codes cv2.waitKey returns, so
# both front ends drive one input handler.
# ---------------------------------------------------------------------------
def _page() -> bytes:
    """The viewer page, with one button per model. Built from MODELS rather
    than hard-coded so appending a detector cannot leave the browser offering
    a stale set of keys the loop no longer maps."""
    digits = "".join(f"'{i + 1}':'{i + 1}'," for i in range(len(MODELS)))
    buttons = "\n".join(
        f" <button onclick=\"k('{i + 1}')\">{i + 1} {n}</button>"
        for i, n in enumerate(MODELS))
    return f"""<!doctype html><meta charset=utf-8><title>BEV detector</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:10px;text-align:center}}
 /* fill the window; --scale sets how many real pixels back that up */
 img{{width:100%;max-width:1900px;height:auto;border:1px solid #333}}
 button{{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;
        padding:6px 14px;margin:2px;font:14px system-ui;cursor:pointer}}
 button:hover{{background:#333}}
 #h{{color:#777;font-size:12px;margin-top:6px}}
</style>
<div>
{buttons}
 <button onclick="k('tab')">Tab cycle</button>
 <button onclick="k('r')">r replane</button>
 <button onclick="k('s')">s snap</button>
</div>
<img src="/stream.mjpg" alt="waiting for the first frame...">
<div id=h>1-{len(MODELS)} switch model, Tab cycles, r re-seeds the plane, s
writes a snapshot on the robot. Ctrl-C in the terminal to quit.</div>
<script>
function k(a){{fetch('/cmd?k='+encodeURIComponent(a))}}
addEventListener('keydown',e=>{{
  const m={{{digits}'r':'r','s':'s','Tab':'tab'}};
  if(m[e.key]!==undefined){{e.preventDefault();k(m[e.key])}}
}});
</script>""".encode()


_PAGE = _page()

# Browser key name -> the code cv2.waitKey would have returned. Single digits,
# so a 10th model would need a different selector (Tab still reaches it).
_KEYMAP = {str(i + 1): ord(str(i + 1)) for i in range(len(MODELS))} | {
    "r": ord("r"), "s": ord("s"), "tab": 9}


class _Control:
    """The one piece of state the HTTP thread and the detection loop share."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._keys: collections.deque[int] = collections.deque(maxlen=16)

    def publish(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg, self._seq = jpeg, self._seq + 1

    def frame(self, since: int) -> tuple[bytes | None, int]:
        """Latest frame if it is newer than ``since`` — so a slow client skips
        stale frames instead of queueing them."""
        with self._lock:
            return (None, since) if self._seq == since else (self._jpeg, self._seq)

    def push_key(self, code: int) -> None:
        with self._lock:
            self._keys.append(code)

    def take_key(self) -> int:
        """Next pending key, or 255 for "nothing" (cv2.waitKey's no-key value)."""
        with self._lock:
            return self._keys.popleft() if self._keys else 255


def _make_handler(ctl: _Control):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a) -> None:      # noqa: A003 - quiet the console
            pass                                # one line per frame otherwise

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:               # noqa: N802 - BaseHTTPRequestHandler API
            path = urllib.parse.urlparse(self.path)
            if path.path == "/":
                return self._send(_PAGE, "text/html; charset=utf-8")
            if path.path == "/cmd":
                k = urllib.parse.parse_qs(path.query).get("k", [""])[0]
                if k in _KEYMAP:
                    ctl.push_key(_KEYMAP[k])
                return self._send(b"ok", "text/plain")
            if path.path == "/frame.jpg":
                # Poll alternative to the MJPEG stream: the latest frame as a
                # plain, complete response (X-Seq names it), or 204 when the
                # caller's ?since= is already the latest. Every browser
                # renders a plain JPEG at once, which is not true of an MJPEG
                # part — live_detect_vlm's page polls this instead because
                # its still frames never showed up in Chrome.
                try:
                    since = int(urllib.parse.parse_qs(path.query).get("since", ["-1"])[0])
                except ValueError:
                    since = -1
                buf, seq = ctl.frame(since)
                if buf is None:
                    self.send_response(204)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(buf)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Seq", str(seq))
                self.end_headers()
                self.wfile.write(buf)
                return
            if path.path != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                # The boundary that CLOSES a part goes out right behind its
                # bytes, not in front of the next part, so a frame published
                # alone is at least a complete part. Firefox renders it then
                # (verified headless, 2026-09-06); Chrome did not, even with
                # further parts following — a script that shows STILL frames
                # should use /frame.jpg polling (live_detect_vlm) rather than
                # this stream. Fine for a continuous live stream like this
                # script's.
                self.wfile.write(b"--frame\r\n")
                while True:
                    buf, seq = ctl.frame(seq)
                    if buf is None:
                        time.sleep(0.01)
                        continue
                    self.wfile.write(b"Content-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(buf)).encode()
                                     + b"\r\n\r\n" + buf + b"\r\n--frame\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                            # browser closed the tab
    return Handler


def serve(port: int, ctl: _Control) -> ThreadingHTTPServer:
    """Start the viewer server on a daemon thread and return it."""
    srv = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(ctl))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="case", choices=list(MODELS),
                    help="which detector to show first (all of them are loaded; "
                         "switch live with 1-N or Tab)")
    ap.add_argument("--conf", type=float, default=0.40, help="min detection confidence")
    ap.add_argument("--plane", type=float, default=None,
                    help="pin the warp plane (m) instead of seeding/locking it from depth")
    ap.add_argument("--angle", type=float, default=24.0, help="head-down align angle")
    ap.add_argument("--scale", type=float, default=1.6,
                    help="display magnification. The BEV canvas is only 840x540 "
                         "(cfg.BEV_PX_PER_M), which is small on a big screen; "
                         "this scales the RENDER (image + overlay + fonts), never "
                         "what the model sees. Costs bandwidth under --serve: "
                         "1.6 is ~2.5x the pixels of 1.0.")
    ap.add_argument("--serve", type=int, metavar="PORT", default=None,
                    help="stream the annotated canvas over HTTP on PORT instead "
                         "of opening a cv2 window, and take the same keys from "
                         "the browser. Needed on this robot: the installed cv2 "
                         "is the headless build, so imshow cannot work.")
    ap.add_argument("--render-every", type=int, default=1,
                    help="draw/imshow only every N frames (detection still runs "
                         "every frame); raise it if a remote display is the bottleneck")
    args = ap.parse_args()

    models = load_models()
    print("loaded:", ", ".join(f"{n}{tuple(m.names.values())}" for n, m in models.items()))
    names = list(MODELS)
    active = args.model
    # Per-model remembered surface height: the plane each target was last
    # measured at. Keyed by model so switching back does not start over.
    locked: dict[str, float] = {}

    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    out = _HERE / cfg.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    with Robot(configs=configs) as robot:
        if not robot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("Warning: camera streams may not be active")
        set_head_pitch(robot, angle=args.angle)

        win = "BEV detector"
        ctl = _Control() if args.serve else None
        if ctl is not None:
            serve(args.serve, ctl)
            print(f"Live BEV detector on http://0.0.0.0:{args.serve}  "
                  f"(buttons/keys in the browser, Ctrl-C here to quit)")
        else:
            try:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                # The installed cv2 is opencv-python-headless, which has no GUI
                # at all — point at the mode that works instead of dying on the
                # first frame.
                raise SystemExit(
                    f"cv2 cannot open a window ({e.err.splitlines()[0].strip()}).\n"
                    f"This build of cv2 is headless — rerun with --serve, e.g.\n"
                    f"    python {Path(__file__).name} --serve 8088 "
                    f"--model {active}") from None
            print(f"Live BEV detector — 1-{len(MODELS)}|Tab model, r replane, s snap, q/Esc quit")
        fps, t, i, disp = 0.0, time.time(), 0, None
        while True:
            t0 = time.time()
            rgb, depth = _get_frame(robot)
            if rgb is None:
                continue
            q_torso, q_head = _joints(robot)
            t1 = time.time()

            # Plane, in order of preference: pinned by the caller, the height
            # this model was last measured at, the depth mode, the model's top
            # face. Only the last is a guess.
            if args.plane is not None:
                plane_z, src = float(args.plane), "pinned"
            elif active in locked:
                plane_z, src = locked[active], f"locked {active}"
            elif depth is not None and (z := seed_plane(depth, rgb.shape,
                                                        q_torso, q_head)[0]) is not None:
                plane_z, src = z, "depth mode"
            else:
                plane_z, src = bev.top_face_z(1), "no depth: fallback"
            plane_z = clamp_plane(plane_z, q_torso, q_head)

            bev_bgr, dets = detect(rgb, depth, q_torso, q_head, models[active],
                                   args.conf, plane_z, MODELS[active][1])
            # Lock onto the best detection's own surface for the next frame.
            if args.plane is None and dets and dets[0]["z_face"] is not None:
                locked[active] = clamp_plane(dets[0]["z_face"], q_torso, q_head)
            t2 = time.time()

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t, 1e-6)
            t = now

            if i % args.render_every == 0:
                # raw frame is only converted when it will actually be shown
                # (nothing detected) — the colour convert is not free at 30 fps.
                raw_bgr = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                           if not dets else None)
                disp = draw(bev_bgr, dets, active, plane_z, src, fps,
                            (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                            scale=args.scale, raw_bgr=raw_bgr)
                if ctl is not None:
                    ok, jpg = cv2.imencode(".jpg", disp,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        ctl.publish(jpg.tobytes())
                else:
                    cv2.imshow(win, disp)
            i += 1
            # One input handler for both front ends: the browser pushes the
            # same codes cv2.waitKey returns.
            key = ctl.take_key() if ctl is not None else cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key < ord("1") + len(names):
                active = names[key - ord("1")]
                print("model ->", active)
            if key == 9:                                    # Tab: cycle
                active = names[(names.index(active) + 1) % len(names)]
                print("model ->", active)
            if key == ord("r"):                             # forget the lock
                locked.pop(active, None)
                print(f"{active}: plane re-seeded from depth")
            if key == ord("s") and disp is not None:
                p = out / f"live_detect_{active}_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(p), disp)
                print(f"saved {p}  (plane {plane_z:.4f}, {src})")
                for d in dets:
                    ee = ("" if d["ee"] is None else
                          f" EE[{d['ee'][1]}]=({d['ee'][0][0]:.3f},"
                          f"{d['ee'][0][1]:+.3f})")
                    raw = "" if _folded(d) is None else f" (obb {d['yaw_raw']:.1f})"
                    print(f"  {d['cls']:9s} conf={d['conf']:.2f} "
                          f"xy=({d['base_xy_z'][0]:.3f},{d['base_xy_z'][1]:+.3f}) "
                          f"yaw={d['yaw']:5.1f}{raw} size={d['dims_m'][0]:.3f}x"
                          f"{d['dims_m'][1]:.3f} z={d['z_face']}{ee}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # --serve has no q key, so Ctrl-C is the way out; unwinding from here
        # still runs the Robot context manager's cleanup.
        print()
