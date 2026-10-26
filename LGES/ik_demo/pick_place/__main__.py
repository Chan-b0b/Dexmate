"""Standalone pick (source) -> hand-driven strafe -> place (target case).

One item, no layer loop, no barcode divert, no seed bin, no auto chassis legs:

    1. operator drives the chassis to the SOURCE station, `d`
    2. BEV detect the source case -> pick the item (case or battery_1)
    3. arm parks clear of the camera WHILE the operator drives to the TARGET
       station, `d`
    4. BEV detect the target case -> place the item on it, end

Pick and place both go straight down onto the detection + the case-frame grab /
slot offset (CASE_GRAB_OFFSET, SLOT_OFFSETS) — 1012: no x/yaw trims, no corner
aim bias, no corner drive, no post-pick wall-relief slide (vision has to put it
there in one go). The --target bin place is chassis_sequence's and still
corner-seats.

The place section is a copy of run_item's target branch (`place_at_target`) plus
place_mover.PlaceMover (copied place / corner-seat / recovery code), so the
place can be redesigned without touching chassis_sequence or suction.py. Measured-z
anchors (ZTracker) are not used: expected heights come from the detected top
face, and the misseat gate is off.

    python -m ik_demo.pick_place --item case --src-layers 3 --tgt-layers 3
    python -m ik_demo.pick_place --item battery1 --src-layers 3 --tgt-layers 3
    python -m ik_demo.pick_place --item case --src-layers 3 --tgt-layers 3 --obb-depth

--obb-depth (1012, --item case only): source AND target planes come from the
depth of the BATTERY TOPS inside the detected box (redetect_on_obb_depth) —
the case is transparent, the batteries are not — instead of the layer model /
the 25px window at the box centre; that plane sets the re-detected xy and the
expected contact z. The batteries seen are the lower case's at both ends, so
the source (an empty case on top of them) adds one layer. The --target bin
place does not use it.

Empty-bin target (case only): the place detects the BIN instead of a case and
seats the case at the empty-bin height (chassis_sequence._place_case_in_bin,
reading ik_demo.config — place_config OVERRIDES do not apply to its aim/seat
z; PlaceMover.place itself still does). --tgt-layers is not needed:

    python -m ik_demo.pick_place --item case --src-layers 1 --target bin

--tgt-layers 0 (case only, 1012): the same empty bin, but placed the pick_place
way (place_at_empty_bin) — bin detection + SEED_BIN_CENTER_OFFSET, straight
down, no corner seat, expected z = bin floor + one layer. A --src-layers 1
source has no batteries under it either, so --obb-depth is skipped for its pick:

    python -m ik_demo.pick_place --item case --src-layers 1 --tgt-layers 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .place_config import (cfg, OBB_DEPTH_BIN_M, OBB_DEPTH_TOL_M, OBB_DEPTH_MIN_PX,
                           OBB_DEPTH_MIN_FRAC, OBB_DEPTH_MAX_SHIFT_M, OBB_DEPTH_INNER_FRAC,
                           BATTERY_TOP_ABOVE_FACE_M,
                           WINDOW_X_M, WINDOW_Y_M, WINDOW_MAX_MOVES,
                           PLACE_RELEASE_ABOVE_M)
from ..arm import connect_robot
from ..config import resolve_poses
from ..drivers import suction_io
from ..go_home import both_arms_home, safe_home
from .place_mover import PlaceMover
from .. import chassis_sequence as cs

# CLI item -> (label, pose key); same pairs as chassis_sequence.ITEMS
ITEM_CHOICES = {"case": ("case", "CASE_PICK"),
                "battery1": ("battery_1", "BAT_SRC_1")}


def obb_plane_z(depth, rgb_shape, q_torso, q_head, det, layers_above: int):
    """Case-face height (base z) from the BATTERY TOPS inside ``det``'s OBB.

    The OBB is a box in the BEV image; its corners go back to the camera image
    through the BEV homography (exact for pixels, whatever plane it was warped
    on), only the inner box (OBB_DEPTH_INNER_FRAC, the battery slots) is kept,
    every valid depth pixel in it is deprojected to base z, and the HIGHEST
    cluster holding >= OBB_DEPTH_MIN_FRAC of them is the battery top. The face
    is BATTERY_TOP_ABOVE_FACE_M below it, + ``layers_above`` LAYER_PITCH_M for
    an empty case nested that many layers above the battery case (see
    place_config). Returns (face_z, n_cluster, n_valid, battery_top_z); the z
    values are None when no cluster is big enough."""
    bev = cs.dcb.bev
    H = np.linalg.inv(bev.build_mapper(q_torso, q_head, det.top_face_z).img_to_bev)
    (cu, cv), (long_px, short_px) = det.bev_center_px, det.dims_px
    a = np.deg2rad(det.base_yaw_deg)       # BEV px angle == base yaw (bev_yaw_to_base)
    ax_l = np.array([np.cos(a), np.sin(a)]) * long_px / 2.0
    ax_s = np.array([-np.sin(a), np.cos(a)]) * short_px / 2.0
    dh, dw = depth.shape[:2]
    rh, rw = rgb_shape[0], rgb_shape[1]

    def poly(scale: float) -> np.ndarray:     # OBB scaled about its centre, depth px
        corners = np.float32([[cu, cv] + scale * (sl * ax_l + ss * ax_s)
                              for sl, ss in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
        img = cv2.perspectiveTransform(corners[None], H)[0]
        return np.round(img * [dw / rw, dh / rh]).astype(np.int32)

    mask = np.zeros((dh, dw), np.uint8)
    cv2.fillPoly(mask, [poly(float(OBB_DEPTH_INNER_FRAC))], 1)
    vs, us = np.nonzero(mask)
    d = depth[vs, us].astype(np.float64)
    ok = np.isfinite(d) & (d > 0.05)
    n_valid = int(ok.sum())
    if n_valid < OBB_DEPTH_MIN_PX:
        return None, 0, n_valid, None
    K = bev.intrinsic_matrix()
    T = bev.cg.zed_left_camera_pose_from_joints(q_torso, q_head)   # base <- cam
    u, v, d = us[ok] * (rw / dw), vs[ok] * (rh / dh), d[ok]
    p = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d])
    z = T[2, :3] @ p + T[2, 3]
    hist, edges = np.histogram(z, np.arange(z.min(), z.max() + 2 * OBB_DEPTH_BIN_M,
                                            OBB_DEPTH_BIN_M))
    centres = 0.5 * (edges[:-1] + edges[1:])
    need = max(OBB_DEPTH_MIN_PX, OBB_DEPTH_MIN_FRAC * n_valid)
    best_n = 0
    for i in np.argsort(centres)[::-1]:          # highest bin first
        if hist[i] < hist[max(i - 1, 0)] or hist[i] < hist[min(i + 1, len(hist) - 1)]:
            continue                             # local peaks only
        inl = np.abs(z - centres[i]) <= OBB_DEPTH_TOL_M
        n = int(inl.sum())
        best_n = max(best_n, n)
        if n >= need:
            top = float(np.median(z[inl]))
            face = top - BATTERY_TOP_ABOVE_FACE_M + layers_above * float(cfg.LAYER_PITCH_M)
            return face, n, n_valid, top
    return None, best_n, n_valid, None


def save_obb_frame(station: str, label: str, rgb, depth, q_torso, q_head, det,
                   fz, n: int, n_valid: int, battery_top=None) -> None:
    """1012 diagnosis: every --obb-depth read is saved as one npz (the frame, the
    joints, the OBB it was read in and what came out) under
    RUN_LOG_DIR/obb_frames/<run stamp>/, so the plane choice can be replayed
    offline with obb_plane_z (the transparent case's face read anywhere in a
    ~15mm range frame to frame, hence the battery-top rule). Never raises."""
    try:
        out = Path(cfg.RUN_LOG_DIR) / "obb_frames" / cs.run_stamp()
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{time.strftime('%H%M%S')}_{station}_{label}.npz"
        np.savez_compressed(
            f, rgb=rgb, depth=depth, q_torso=np.asarray(q_torso), q_head=np.asarray(q_head),
            det_bev_center_px=np.asarray(det.bev_center_px), det_dims_px=np.asarray(det.dims_px),
            det_base_yaw_deg=det.base_yaw_deg, det_base_xy=np.asarray(det.base_xy),
            det_top_face_z=det.top_face_z, det_conf=det.conf,
            plane_z=np.nan if fz is None else fz, n_cluster=n, n_valid=n_valid,
            battery_top_z=np.nan if battery_top is None else battery_top)
        logger.info("[{}] OBB depth frame -> {}", label, f)
    except Exception as e:  # noqa: BLE001 — a diagnostic save must not stop a run
        logger.warning("[{}] OBB depth frame save failed ({})", label, e)


def redetect_on_obb_depth(bot, label: str, layers: int, det, station: str,
                          layers_above: int):
    """--obb-depth: measure the plane inside ``det``'s OBB (from the battery
    tops, ``layers_above`` = layers of empty case above them), re-detect on it (the
    warp plane sets the detected xy) — the new detection's top_face_z is that
    plane, so it also sets the expected contact z. None = no plane / no
    re-detection (the caller hands the operator the keyboard). Returns
    (detection, battery_top_z): the battery top is what the cup meets — the
    picked empty case and the placed one both rest ON the lower case's
    batteries (1012: place contact ee z == pick contact, both = battery top +
    SUCTION_LENGTH_M within 2mm) — so it, not the xy plane, sets the expected z."""
    rgb = cs._head_rgb(bot, fresh=False)
    depth = bot.sensors.head_camera.get_depth()
    if rgb is None or depth is None:
        logger.warning("[{}] OBB depth: no frame (rgb={}, depth={})",
                       label, rgb is not None, depth is not None)
        return None, None
    q_torso, q_head = cs._joints(bot)
    fz, n, n_valid, top = obb_plane_z(depth, rgb.shape, q_torso, q_head, det, layers_above)
    save_obb_frame(station, label, rgb, depth, q_torso, q_head, det, fz, n, n_valid, top)
    if fz is None:
        logger.warning("[{}] OBB depth: no battery-top cluster (best {} of {} valid px)",
                       label, n, n_valid)
        return None, None
    logger.info("[{}] OBB depth: battery top {:.4f} ({} of {} px) -> case face z={:.4f} "
                "({} layer(s) above) vs model {:.4f} ({:+.1f}mm) — re-detecting on it",
                label, top, n, n_valid, fz, layers_above, det.top_face_z,
                (fz - det.top_face_z) * 1000.0)
    fresh = cs.detect(bot, layers, fz)
    if fresh is None or not fresh.found:
        logger.warning("[{}] re-detect on the OBB-depth plane failed", label)
        return None, None
    shift = float(np.hypot(fresh.base_xy[0] - det.base_xy[0],
                           fresh.base_xy[1] - det.base_xy[1]))
    if shift > OBB_DEPTH_MAX_SHIFT_M:
        logger.warning("[{}] OBB-depth re-detect jumped {:.0f}mm ({:.3f},{:+.3f}) -> "
                       "({:.3f},{:+.3f}) — a different box, not the same case moved",
                       label, shift * 1000.0, det.base_xy[0], det.base_xy[1],
                       fresh.base_xy[0], fresh.base_xy[1])
        return None, None
    logger.info("[{}] OBB-depth re-detect: xy ({:.3f},{:+.3f}) -> ({:.3f},{:+.3f})",
                label, det.base_xy[0], det.base_xy[1], fresh.base_xy[0], fresh.base_xy[1])
    return fresh, top


def _into_window(bot, label: str, xy, moves: int) -> "str":
    """Drive the chassis so the detected case centre ``xy`` lands in the
    WINDOW_X_M / WINDOW_Y_M window. Returns "ok" (already inside), "moved"
    (chassis moved -> the caller re-detects) or "stuck" (still outside after
    WINDOW_MAX_MOVES moves, ``moves`` = moves so far -> the caller hands the
    operator the keyboard). Moves go to the window CENTRE, not its edge."""
    from ..move_chassis import move_diagonal
    x, y = float(xy[0]), float(xy[1])
    out_x = WINDOW_X_M is not None and abs(x - WINDOW_X_M[0]) > WINDOW_X_M[1]
    out_y = WINDOW_Y_M is not None and abs(y - WINDOW_Y_M[0]) > WINDOW_Y_M[1]
    if not (out_x or out_y):
        return "ok"
    # one axis outside -> BOTH axes go to the window centre (an axis that is
    # already inside is not left where it is)
    dx = x - WINDOW_X_M[0] if WINDOW_X_M is not None else 0.0   # +: too far ahead -> close on it
    dy = y - WINDOW_Y_M[0] if WINDOW_Y_M is not None else 0.0   # +: to the left -> strafe left
    if moves >= WINDOW_MAX_MOVES:
        logger.warning("[{}] case ({:.3f},{:+.3f}) still outside the window after {} moves",
                       label, x, y, moves)
        return "stuck"
    lim = float(cfg.CHASSIS_ADJUST_MAX_TRANSLATE_M)
    dx, dy = float(np.clip(dx, -lim, lim)), float(np.clip(dy, -lim, lim))
    logger.info("[{}] case ({:.3f},{:+.3f}) outside the window -> dx {:+.3f} m, dy {:+.3f} m",
                label, x, y, dx, dy)
    move_diagonal(bot, dx, dy)     # both axes at once, one straight line
    return "moved"


def pick_at_source(bot, mover: PlaceMover, label: str, pose_key: str,
                   src_layers: int, obb_depth: bool = False) -> "float | None":
    """Detect the source case and pick `pose_key`. Returns the case yaw (rad,
    as picked) on success; a failure homes the arms and returns None (nothing
    held)."""
    if obb_depth and src_layers <= 1:
        # a layer-1 case sits on the bin floor: no batteries under it to read,
        # the model plane (bin floor + 1 layer) sets the pick z
        logger.info("[{}] src layers {} — no batteries below, OBB depth off for the "
                    "pick (bin floor plane)", label, src_layers)
        obb_depth = False
    win_moves = 0
    while True:
        det = cs.detect(bot, src_layers)
        if det is None or not det.found:
            logger.error("[{}] source detect failed — arms home", label)
            cs._arms_home(bot, mover)
            return None
        plane = bat_top = None
        if obb_depth:
            fresh, bat_top = redetect_on_obb_depth(bot, label, src_layers, det, "source", 1)
            if fresh is None:
                logger.warning("[{}] source OBB depth failed — adjust the chassis (f/b/l/r), "
                               "`d` to re-detect + retry, `q` to give up", label)
                if not cs._manual_strafe(bot, "adjust"):
                    cs._arms_home(bot, mover)
                    return None
                continue
            det, plane = fresh, fresh.top_face_z
        det = cs._refine_det(bot, src_layers, det, plane)
        win = _into_window(bot, label, det.base_xy, win_moves)
        if win == "moved":
            win_moves += 1
            continue
        if win == "stuck":
            if not cs._manual_strafe(bot, "adjust"):
                cs._arms_home(bot, mover)
                return None
            win_moves = 0
            continue
        logger.info("[{}] source case @ base xy=({:.3f},{:+.3f}) yaw={:.1f}deg conf={:.2f}",
                    label, det.base_xy[0], det.base_xy[1], det.base_yaw_deg, det.conf)
        center = cs._center_from_det(det)
        pick_pose = resolve_poses(center)[pose_key]
        if cs.descent_reachable(mover, pick_pose):
            break
        logger.warning("[{}] pick pose out of reach — adjust the chassis (f/b/l/r), "
                       "`d` to re-detect + retry, `q` to give up", label)
        if not cs._manual_strafe(bot, "adjust"):
            cs._arms_home(bot, mover)
            return None
    exp_z = (det.top_face_z if bat_top is None else bat_top) + cfg.SUCTION_LENGTH_M
    res = mover.pick(pick_pose, expected_z=exp_z, lift_to_clear=True,
                     lift_creep_out_m=(cfg.DESCENT_CREEP_GAP_M if label == "case" else 0.0))
    if not res.success:
        logger.error("[{}] pick failed: {} — arms home", label, res.reason)
        cs._arms_home(bot, mover)
        return None
    return center[3]


def _apply_bin_config() -> None:
    # _place_case_in_bin / _detect_bin_xy read chassis_sequence's config, not place_config,
    # so the place_config values for the bin never reached them: the bin plane stayed on the
    # old 0.6138 (seat-minus-bin offset grew with bin distance, 107->145mm). Apply them here.
    for name in ("DIVERT_BIN_PLANE_Z_M", "SEED_BIN_CENTER_OFFSET"):
        new = getattr(cfg, name)
        if getattr(cs.cfg, name) != new:
            logger.warning("bin place: {} {} -> {} (place_config)", name, getattr(cs.cfg, name), new)
            setattr(cs.cfg, name, new)


def place_in_bin(bot, mover: PlaceMover, label: str, source_yaw: float) -> bool:
    """Place the held case into the EMPTY bin from a bin detection. Returns
    False if the operator stopped with the case still held."""
    _apply_bin_config()
    pres = cs._place_case_in_bin(bot, mover, label, source_yaw, 0, auto=False)
    if pres is None:
        return False
    if not getattr(pres, "success", True):
        logger.warning("[{}] bin place failed ({}) — operator resolved it at the gate",
                       label, pres.reason)
    return True


def place_at_empty_bin(bot, mover: PlaceMover, label: str, source_yaw: float) -> bool:
    """--tgt-layers 0: place the held case into the EMPTY bin the same way
    place_at_target places onto a case — straight down, no corner aim bias /
    corner drive. xy = bin detection + SEED_BIN_CENTER_OFFSET, yaw = the bin's
    long-axis yaw (the case seats long side along the bin's; ``source_yaw``, the
    yaw it was picked at, is only logged), expected z = bin floor + one layer +
    SUCTION_LENGTH_M (no batteries to read a plane from). The centre goes
    through the same window as a case place. Returns False if the case is
    still held when it stops."""
    _apply_bin_config()
    exp_z = cfg.FLOOR_Z_BASE_M + cfg.LAYER_PITCH_M + cfg.SUCTION_LENGTH_M
    win_moves = 0
    while True:
        bxy = cs._detect_bin_xy(bot, with_yaw=True)
        if bxy is None:
            logger.warning("[{}] bin NOT detected — adjust the chassis (f/b/l/r), `d` to "
                           "re-detect + retry, `q` to stop (case still held)", label)
            if not cs._manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the bin (case still held)", label)
                return False
            continue
        xy = (bxy[0] + cfg.SEED_BIN_CENTER_OFFSET[0], bxy[1] + cfg.SEED_BIN_CENTER_OFFSET[1])
        win = _into_window(bot, label, xy, win_moves)
        if win == "moved":
            win_moves += 1
            continue
        if win == "stuck":
            if not cs._manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the bin (case still held)", label)
                return False
            win_moves = 0
            continue
        bin_yaw = float(np.deg2rad(bxy[2]))
        place_pose = resolve_poses((*xy, exp_z, bin_yaw))["CASE_PICK"]
        logger.info("[{}] empty bin ({:.3f},{:+.3f}) yaw {:+.1f}deg (picked at {:+.1f}) -> "
                    "place aim ({:.4f},{:+.4f}) ee yaw {:+.1f}deg, expected ee_z {:.4f} "
                    "(bin floor + 1 layer)", label, bxy[0], bxy[1], bxy[2],
                    float(np.rad2deg(source_yaw)), place_pose[0], place_pose[1],
                    float(np.rad2deg(place_pose[5])), exp_z)
        if not cs.descent_reachable(mover, place_pose):
            logger.warning("[{}] bin place pose out of reach — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to stop (case still held)", label)
            if not cs._manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the bin (case still held)", label)
                return False
            continue
        pres = mover.place(place_pose, expected_z=exp_z, lift_to_clear=True,
                           release_above_m=PLACE_RELEASE_ABOVE_M)
        break
    if pres is not None and not getattr(pres, "success", True):
        if pres.reason == "unreachable" and pres.contact_ee_z is None:
            logger.error("[{}] bin place failed: {} (case still held)", label, pres.reason)
            return False
        logger.warning("[{}] bin place failed ({}) — operator resolved it at the gate",
                       label, pres.reason)
    return True


def place_at_target(bot, mover: PlaceMover, label: str, pose_key: str,
                    tgt_layers: int, obb_depth: bool = False) -> bool:
    """Detect the target case and place the held item on it. Returns False if
    the item is still held when it stops."""
    redetects = 0
    win_moves = 0
    while True:
        # every round measures its plane afresh: a round ends in a chassis move
        # (window / operator adjust), and the depth read must come from the
        # frame the pose is computed from
        tgt_plane = bat_top = None
        tdet = cs.detect(bot, tgt_layers)
        if tdet is None or not tdet.found:
            # a case MUST be at the target — a miss is a detection failure, so
            # re-detect once, then hand the operator the keyboard
            if redetects < 1:
                redetects += 1
                logger.warning("[{}] target case NOT detected — re-detecting once", label)
                continue
            logger.warning("[{}] target case NOT detected — adjust the chassis "
                           "(f/b/l/r), `d` to re-detect + retry, `q` to stop "
                           "(item still held)", label)
            if not cs._manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the place station (item still held)", label)
                return False
            continue

        logger.info("[{}] target case found -> aligned place", label)
        if obb_depth:
            fresh, bat_top = redetect_on_obb_depth(bot, label, tgt_layers, tdet, "target", 0)
            if fresh is None:
                logger.warning("[{}] target OBB depth failed — adjust the chassis "
                               "(f/b/l/r), `d` to re-detect + retry, `q` to stop "
                               "(item still held)", label)
                if not cs._manual_strafe(bot, "adjust"):
                    logger.error("[{}] stopping at the place station (item still held)", label)
                    return False
                continue
            tgt_plane, tdet = fresh.top_face_z, fresh
        else:
            # the detection above was warped on the MODEL plane, which sat
            # 35-48mm off at the target (0914-0915); measure the top face with
            # the depth camera and re-detect on it
            model = cs.dcb.bev.top_face_z(tgt_layers)
            fz, n_px, spread = cs._depth_face_z(bot, label,
                                                cs._center_from_det(tdet)[:2], model)
            if fz is None:
                logger.warning("[{}] target face depth: only {} valid px — keeping "
                               "the model plane", label, n_px)
            else:
                logger.info("[{}] target face depth: z={:.4f} vs model {:.4f} "
                            "({:+.1f}mm, {} px, relief {:.0f}mm) — re-detecting on it",
                            label, fz, model, (fz - model) * 1000.0, n_px, spread * 1000.0)
                fresh = cs.detect(bot, tgt_layers, fz)
                if fresh is None or not fresh.found:
                    logger.warning("[{}] re-detect on the depth plane failed — keeping "
                                   "the model-plane detection", label)
                else:
                    tgt_plane, tdet = fz, fresh
        tdet = cs._refine_det(bot, tgt_layers, tdet, tgt_plane)
        win = _into_window(bot, label, tdet.base_xy, win_moves)
        if win == "moved":
            win_moves += 1
            continue
        if win == "stuck":
            if not cs._manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the place station (item still held)", label)
                return False
            win_moves = 0
            continue
        tc = cs._center_from_det(tdet)
        place_pose = resolve_poses(tc)[pose_key]
        # --obb-depth: the held case lands ON the batteries (see redetect_on_obb_depth)
        exp_z = (tdet.top_face_z if bat_top is None else bat_top) + cfg.SUCTION_LENGTH_M
        # 1012: the pose is the detection + the case-frame grab/slot offset only
        # — no x/yaw trims, no corner aim bias, no corner drive
        logger.info("[{}] place aim ({:.4f},{:+.4f}) yaw {:+.1f}deg, expected ee_z {:.4f}",
                    label, place_pose[0], place_pose[1],
                    float(np.rad2deg(place_pose[5])), exp_z)
        if not cs.descent_reachable(mover, place_pose):
            logger.warning("[{}] place pose out of reach — adjust the chassis (f/b/l/r), "
                           "`d` to re-detect + retry, `q` to stop (item still held)", label)
            if not cs._manual_strafe(bot, "adjust"):
                logger.error("[{}] stopping at the place station (item still held)", label)
                return False
            continue

        pres = mover.place(place_pose, expected_z=exp_z, lift_to_clear=True,
                           hover_m=(None if label == "case"
                                    else cfg.BATTERY_PLACE_HOVER_M),
                           release_above_m=PLACE_RELEASE_ABOVE_M)
        break
    if pres is not None and not getattr(pres, "success", True):
        if pres.reason == "unreachable" and pres.contact_ee_z is None:
            logger.error("[{}] place failed: {} (part still held)", label, pres.reason)
            return False
        logger.warning("[{}] place failed ({}) — operator resolved it at the gate",
                       label, pres.reason)
    return True


def _ask_next(args) -> "bool | None":
    """After a place: Enter repeats with the same settings, `q` stops, or type
    changes -- `battery1`, `3 2` (src tgt layers), `case 3 2`
    -- applied to ``args`` in place. Returns None to stop."""
    while True:
        line = input(f"next pick: item={args.item} src={args.src_layers} tgt={args.tgt_layers} "
                     f"[Enter=same, q=stop, e.g. `battery1 3 2`]: ").strip().lower()
        if line in ("q", "quit"):
            return None
        item, nums, ok = args.item, [], True
        for tok in line.split():
            if tok in ITEM_CHOICES:
                item = tok
            elif tok.lstrip("-").isdigit():
                nums.append(int(tok))
            else:
                ok = False
        if not ok or len(nums) > 2:
            print("could not parse; try `battery1 3 2`, `3 2` or Enter")
            continue
        if nums:
            args.src_layers = nums[0]
            if len(nums) == 2:
                args.tgt_layers = nums[1]
        if item == "battery1" and (args.target == "bin" or args.tgt_layers == 0):
            print("--target bin / tgt 0 (empty bin) works for the case only")
            continue
        args.item = item
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", choices=sorted(ITEM_CHOICES), required=True)
    ap.add_argument("--src-layers", type=int, required=True,
                    help="physical layers in the source stack")
    ap.add_argument("--tgt-layers", type=int, default=None,
                    help="physical layers in the target stack (the case to place on); "
                         "required for --target case")
    ap.add_argument("--target", choices=("case", "bin"), default="case",
                    help="case: place on a case at the target (default); "
                         "bin: place the case into an EMPTY bin (--item case only)")
    ap.add_argument("--obb-depth", action="store_true",
                    help="(--item case) measure each case's plane from the depth of the "
                         "battery tops inside its detected box, re-detect on it for xy and "
                         "use it as the expected contact z; a failed read stops for the "
                         "operator. The layer counts still give the first detection's plane")
    args = ap.parse_args()
    if args.target == "bin" and args.item != "case":
        ap.error("--target bin works for --item case only")
    if args.obb_depth and args.item != "case":
        ap.error("--obb-depth works for --item case only (a battery source/target slot "
                 "may hold no battery to read)")
    if args.target == "case" and args.tgt_layers is None:
        ap.error("--tgt-layers is required for --target case")
    if args.tgt_layers == 0 and args.item != "case":
        ap.error("--tgt-layers 0 (empty bin) works for --item case only")
    label, pose_key = ITEM_CHOICES[args.item]

    # import dexcontrol BEFORE the log sinks: its __init__ calls
    # logger.configure(handlers=...), which drops every sink added earlier
    # (1012: the run log file stopped after its first line)
    from dexcontrol.core.config import get_robot_config
    cs.setup_logging()
    if input(f"MOVES THE REAL ARMS + SUCTION + CHASSIS. Pick {label} (src {args.src_layers}) "
             f"-> place ({'EMPTY BIN' if args.target == 'bin' or args.tgt_layers == 0 else f'tgt {args.tgt_layers}'}). Connect + safe-home? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with connect_robot(configs) as bot:
        for part in (bot.left_arm, bot.right_arm, bot.chassis, bot.torso, bot.head):
            part.set_subscription_policy("always_on", recursive=True)
        bot.sensors.head_camera._streams["left_rgb"].set_subscription_policy("always_on")
        bot.sensors.head_camera.wait_for_active(timeout=5.0)
        cg = cs.dcb.bev.cg
        logger.info("intrinsics from {}: cx={:.2f} cy={:.2f}",
                    "camera info" if cg.load_intrinsics(bot) else "FALLBACK", cg.ZED_CX, cg.ZED_CY)
        with PlaceMover(bot) as m:
            release = m.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not m.ensure_ready(release_estop=release):
                raise SystemExit("arm not ready")
            both_arms_home(bot, left=m)
            cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
            time.sleep(2.0)

            while True:
                ok = False
                try:
                    logger.info("position the chassis at the SOURCE station, then `d`")
                    if not cs._manual_strafe(bot, "source"):
                        logger.error("aborted (`q`)")
                        return
                    src_yaw = pick_at_source(bot, m, label, pose_key, args.src_layers,
                                             args.obb_depth)
                    if src_yaw is None:
                        return
                    logger.info("position the chassis at the {} station, then `d`",
                                "BIN" if args.target == "bin" or args.tgt_layers == 0 else "TARGET")
                    arrived = []
                    cs._park_during_legs(label, [lambda: cs._view_park(m, label)],
                                         lambda: arrived.append(cs._manual_strafe(bot, "target")))
                    if not arrived[0]:
                        logger.error("aborted (`q`) — {} still held", label)
                        return
                    if args.target == "bin":
                        ok = place_in_bin(bot, m, label, src_yaw)
                    elif args.tgt_layers == 0:
                        ok = place_at_empty_bin(bot, m, label, src_yaw)
                    else:
                        ok = place_at_target(bot, m, label, pose_key, args.tgt_layers,
                                             args.obb_depth)
                finally:
                    if ok:
                        safe_home(m)
                    logger.info("pick_place {}", "OK" if ok else "STOPPED (check what is held)")
                if not ok:
                    return
                nxt = _ask_next(args)
                if nxt is None:
                    return
                label, pose_key = ITEM_CHOICES[args.item]
                cs.set_head_pitch(bot, angle=24.0, tol_deg=0.3)
                time.sleep(2.0)


if __name__ == "__main__":
    main()
