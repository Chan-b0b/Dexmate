#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Measure the height of a planar surface in front of the robot, in base frame.

Pipeline:
    1. Grab N head-camera frames (left_rgb + depth) and take a per-pixel median
       to suppress stereo noise.
    2. Back-project the depth ROI into 3D points using the left camera
       intrinsics from ``sensors/head_camera/info``.
    3. Transform the points into the robot base frame using forward kinematics
       (``dexmotion.MotionManager.fk``) on the current torso/head joint state.
    4. Fit a plane with RANSAC + SVD in base frame. The inlier centroid's z is
       the plane height; the normal's tilt from base +z is the quality check.

The camera frame name comes from the URDF and differs per robot model, so run
``--list-frames`` first to find it:

    python examples/advanced_examples/measure_plane_height.py --list-frames

Then measure (drag a box over the surface in the RGB window, press Enter):

    python examples/advanced_examples/measure_plane_height.py \
        --camera-frame head_camera_left_optical_frame

Headless, with an explicit ROI and a floor cross-check:

    python examples/advanced_examples/measure_plane_height.py \
        --camera-frame head_camera_left_optical_frame \
        --roi 700 500 1200 800 --floor-roi 800 950 1100 1070

IMPORTANT - frame convention: if the FK frame you pass is a ROS-style camera
*link* frame (x forward, y left, z up) rather than an *optical* frame (x right,
y down, z forward), pass ``--link-frame-convention`` so the optical->link
rotation is applied. Use ``--floor-roi`` to verify: the reported floor height
must come out near 0 and its tilt near 0 degrees. If it does not, the frame name
or the convention flag is wrong.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", "")

import matplotlib

try:
    matplotlib.use("TkAgg")
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger

from dexcontrol.core.config import get_robot_config
from dexcontrol.robot import Robot

# Rotation mapping optical convention (x right, y down, z forward) to
# REP-103 link convention (x forward, y left, z up).
R_LINK_FROM_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def extract_intrinsics(
    info: dict[str, Any], depth_shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Pull fx, fy, cx, cy out of a dexsensor camera-info payload.

    The payload schema is defined by dexsensor, not dexcontrol, so this searches
    the nested dict for either explicit fx/fy/cx/cy keys or a 3x3 camera matrix,
    preferring the left / depth-aligned branch. Intrinsics are rescaled if the
    payload's resolution differs from the received depth image.

    Args:
        info: Raw payload from ``BaseCameraSensor.get_camera_info()``.
        depth_shape: (height, width) of the depth image actually received.

    Returns:
        (fx, fy, cx, cy) scaled to the depth image resolution.

    Raises:
        ValueError: If no intrinsics could be located in the payload.
    """
    candidates: list[tuple[int, tuple[float, float, float, float], Any, Any]] = []

    def score(path: str) -> int:
        p = path.lower()
        s = 0
        if "left" in p:
            s += 3
        if "depth" in p:
            s += 2
        if "rect" in p:
            s += 1
        if "right" in p:
            s -= 5
        return s

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return

        lower = {str(k).lower(): v for k, v in node.items()}

        # Form 1: explicit scalar keys.
        if all(k in lower for k in ("fx", "fy", "cx", "cy")):
            try:
                k = (
                    float(lower["fx"]),
                    float(lower["fy"]),
                    float(lower["cx"]),
                    float(lower["cy"]),
                )
                candidates.append(
                    (score(path), k, lower.get("width"), lower.get("height"))
                )
            except (TypeError, ValueError):
                pass

        # Form 2: a camera matrix under a conventional key.
        for key in ("k", "camera_matrix", "intrinsic_matrix", "intrinsics"):
            mat = lower.get(key)
            if mat is None:
                continue
            if isinstance(mat, (dict, str)):
                continue
            try:
                flat = np.asarray(mat, dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            if flat.size == 9:
                candidates.append(
                    (
                        score(path) + 1,
                        (flat[0], flat[4], flat[2], flat[5]),
                        lower.get("width"),
                        lower.get("height"),
                    )
                )

        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else str(key))

    walk(info, "")

    if not candidates:
        raise ValueError(
            "Could not find intrinsics in the camera info payload. Inspect it "
            "with examples/troubleshooting/dump_camera_info.py and pass "
            "--fx/--fy/--cx/--cy manually."
        )

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, (fx, fy, cx, cy), info_w, info_h = candidates[0]

    height, width = depth_shape
    if info_w and info_h:
        sx = width / float(info_w)
        sy = height / float(info_h)
        if not (0.99 < sx < 1.01 and 0.99 < sy < 1.01):
            logger.warning(
                f"Camera info resolution {info_w}x{info_h} differs from depth "
                f"{width}x{height}; scaling intrinsics by ({sx:.3f}, {sy:.3f})"
            )
            fx, cx = fx * sx, cx * sx
            fy, cy = fy * sy, cy * sy

    return fx, fy, cx, cy


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def back_project(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    roi: tuple[int, int, int, int] | None,
    z_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a depth ROI into 3D points in the camera optical frame.

    Args:
        depth: (H, W) depth image in meters.
        intrinsics: (fx, fy, cx, cy) matching the depth resolution.
        roi: (x0, y0, x1, y1) pixel box, or None for the whole image.
        z_range: (min, max) valid depth in meters.

    Returns:
        Tuple of (points (N, 3) float64, pixel_indices (N, 2) as (v, u)).
    """
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    v_grid, u_grid = np.mgrid[0:height, 0:width]

    valid = np.isfinite(depth) & (depth > z_range[0]) & (depth < z_range[1])
    if roi is not None:
        x0, y0, x1, y1 = roi
        box = np.zeros((height, width), dtype=bool)
        box[max(y0, 0) : y1, max(x0, 0) : x1] = True
        valid &= box

    z = depth[valid].astype(np.float64)
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)

    points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
    pixels = np.stack([v_grid[valid], u_grid[valid]], axis=1)
    return points, pixels


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
    max_samples: int = 20000,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane to a point set with RANSAC, refined by SVD on the inliers.

    Args:
        points: (N, 3) array of 3D points.
        threshold: Inlier distance threshold in meters.
        iterations: Number of RANSAC hypotheses.
        seed: RNG seed for reproducibility.
        max_samples: Subsample size used for hypothesis scoring.

    Returns:
        Tuple of (unit normal (3,), plane offset d such that n.p + d = 0,
        boolean inlier mask over ``points``).

    Raises:
        ValueError: If fewer than 3 points are supplied or no hypothesis is
            found.
    """
    if len(points) < 3:
        raise ValueError(f"Need at least 3 points to fit a plane, got {len(points)}")

    rng = np.random.default_rng(seed)

    # Score hypotheses on a subsample; the final refit uses every point.
    if len(points) > max_samples:
        sample = points[rng.choice(len(points), max_samples, replace=False)]
    else:
        sample = points

    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = -1

    for _ in range(iterations):
        p0, p1, p2 = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = float(-normal @ p0)
        count = int(np.count_nonzero(np.abs(sample @ normal + offset) < threshold))
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset

    if best_normal is None:
        raise ValueError("RANSAC failed to find any valid plane hypothesis")

    # Refine on all inliers of the best hypothesis (two passes).
    normal, offset = best_normal, best_offset
    inliers = np.abs(points @ normal + offset) < threshold
    for _ in range(2):
        subset = points[inliers]
        if len(subset) < 3:
            break
        centroid = subset.mean(axis=0)
        normal = np.linalg.svd(subset - centroid, full_matrices=False)[2][-1]
        offset = float(-normal @ centroid)
        inliers = np.abs(points @ normal + offset) < threshold

    return normal, offset, inliers


def analyze_plane(
    points_base: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Fit a plane in base frame and summarize its height and quality.

    Args:
        points_base: (N, 3) points already expressed in the robot base frame.
        threshold: RANSAC inlier threshold in meters.
        iterations: RANSAC iteration count.
        seed: RNG seed.

    Returns:
        Dictionary with height_m, tilt_deg, rms_mm, inlier_ratio, n_inliers,
        normal, centroid and the inlier mask.
    """
    normal, offset, inliers = fit_plane_ransac(
        points_base, threshold=threshold, iterations=iterations, seed=seed
    )

    # Point the normal up so the tilt reading is unambiguous.
    if normal[2] < 0:
        normal, offset = -normal, -offset

    subset = points_base[inliers]
    centroid = subset.mean(axis=0)
    residual = subset @ normal + offset
    tilt = np.degrees(np.arccos(float(np.clip(abs(normal[2]), 0.0, 1.0))))

    return {
        "height_m": float(centroid[2]),
        "tilt_deg": float(tilt),
        "rms_mm": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "inlier_ratio": float(inliers.mean()),
        "n_inliers": int(inliers.sum()),
        "n_points": int(len(points_base)),
        "normal": normal,
        "centroid": centroid,
        "inliers": inliers,
    }


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def grab_frames(
    camera, n_frames: int, timeout: float
) -> tuple[np.ndarray, np.ndarray]:
    """Collect RGB and median-filtered depth from the head camera.

    Args:
        camera: A ``ZedCameraSensor`` instance.
        n_frames: Number of depth frames to median together.
        timeout: Overall acquisition timeout in seconds.

    Returns:
        Tuple of (left_rgb (H, W, 3) uint8, depth (H, W) float64 in meters).

    Raises:
        RuntimeError: If not enough frames arrive before the timeout.
    """
    import time

    depths: list[np.ndarray] = []
    rgb: np.ndarray | None = None
    deadline = time.time() + timeout
    logged_dtype = False

    while len(depths) < n_frames and time.time() < deadline:
        obs = camera.get_obs(obs_keys=["left_rgb", "depth"])
        depth = obs.get("depth")
        if obs.get("left_rgb") is not None:
            rgb = obs["left_rgb"]
        if depth is not None:
            depth = np.asarray(depth)
            if not logged_dtype:
                logger.info(f"depth stream dtype={depth.dtype} shape={depth.shape}")
                logged_dtype = True
            # PNG-coded depth can arrive as uint16 millimeters.
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float64) / 1000.0
            else:
                depth = depth.astype(np.float64)
            depth[depth <= 0] = np.nan
            depths.append(depth)
        time.sleep(0.02)

    if not depths or rgb is None:
        raise RuntimeError(
            f"Timed out: got {len(depths)}/{n_frames} depth frames, "
            f"rgb={'ok' if rgb is not None else 'missing'}"
        )
    if len(depths) < n_frames:
        logger.warning(f"Only collected {len(depths)}/{n_frames} depth frames")

    with np.errstate(invalid="ignore"):
        depth_median = np.nanmedian(np.stack(depths, axis=0), axis=0)
    return rgb, depth_median


def select_roi(image: np.ndarray, title: str) -> tuple[int, int, int, int] | None:
    """Let the user drag a box on an image.

    Args:
        image: RGB image to display.
        title: Prompt shown in the window title.

    Returns:
        (x0, y0, x1, y1) or None if the user skipped the selection.
    """
    from matplotlib.widgets import RectangleSelector

    result: dict[str, tuple[int, int, int, int]] = {}
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(image)
    ax.set_title(f"{title}\ndrag a box, then press Enter  (q = skip)")

    def on_select(click, release) -> None:
        if click.xdata is None or release.xdata is None:
            return
        x0, x1 = sorted((int(click.xdata), int(release.xdata)))
        y0, y1 = sorted((int(click.ydata), int(release.ydata)))
        result["roi"] = (x0, y0, x1, y1)
        ax.set_title(f"ROI = {result['roi']}\nEnter = accept, q = skip")
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], minspanx=5, minspany=5, interactive=True
    )

    def on_key(event) -> None:
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key in ("q", "escape"):
            result.pop("roi", None)
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    del selector  # keep the reference alive until the window closes

    return result.get("roi")


def show_result(
    rgb: np.ndarray,
    pixels: np.ndarray,
    inliers: np.ndarray,
    depth_shape: tuple[int, int],
    title: str,
) -> None:
    """Overlay the plane inlier pixels on the RGB image for visual checking.

    Args:
        rgb: RGB image to draw on.
        pixels: (N, 2) pixel coordinates as (v, u) in depth-image space.
        inliers: Boolean mask over ``pixels``.
        depth_shape: (height, width) of the depth image the pixels came from.
        title: Window title.
    """
    overlay = rgb.copy()
    inlier_px = pixels[inliers]

    # Depth and RGB can differ in resolution; rescale into RGB pixel space.
    if rgb.shape[:2] != depth_shape:
        sy = rgb.shape[0] / depth_shape[0]
        sx = rgb.shape[1] / depth_shape[1]
        inlier_px = np.stack(
            [
                np.clip((inlier_px[:, 0] * sy).astype(int), 0, rgb.shape[0] - 1),
                np.clip((inlier_px[:, 1] * sx).astype(int), 0, rgb.shape[1] - 1),
            ],
            axis=1,
        )

    overlay[inlier_px[:, 0], inlier_px[:, 1]] = (
        0.45 * overlay[inlier_px[:, 0], inlier_px[:, 1]]
        + 0.55 * np.array([0, 255, 0])
    ).astype(overlay.dtype)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(overlay)
    ax.set_title(f"{title}\n(green = plane inliers)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #
def build_motion_manager(bot: Robot):
    """Create a MotionManager seeded with the robot's current joint state.

    Args:
        bot: Connected ``Robot`` instance.

    Returns:
        A ``dexmotion.motion_manager.MotionManager``.
    """
    from dexmotion.motion_manager import MotionManager

    components = ["left_arm", "right_arm", "head"]
    if bot.has_component("torso"):
        components.insert(2, "torso")

    joint_pos_dict = bot.get_joint_pos_dict(component=components)
    logger.info(f"FK seeded with components: {components}")

    # URDF loading blocks for a while; keep the watchdog from tripping.
    paused = False
    try:
        bot.heartbeat.pause()
        paused = True
    except Exception as exc:  # pragma: no cover - depends on robot build
        logger.debug(f"Could not pause heartbeat: {exc}")

    try:
        return MotionManager(initial_joint_configuration_dict=joint_pos_dict)
    finally:
        if paused:
            try:
                bot.heartbeat.resume()
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Could not resume heartbeat: {exc}")


def list_frame_names(motion_manager) -> list[str]:
    """Return every frame name in the loaded pinocchio model."""
    return [frame.name for frame in motion_manager.pin_robot.model.frames]


def get_camera_to_base(motion_manager, frame_name: str) -> np.ndarray:
    """Compute the 4x4 transform from the camera frame to the URDF root frame.

    Args:
        motion_manager: Initialized ``MotionManager``.
        frame_name: URDF frame name of the camera.

    Returns:
        4x4 homogeneous transform T_base_camera.

    Raises:
        KeyError: If the frame is unknown to the model.
    """
    available = list_frame_names(motion_manager)
    if frame_name not in available:
        hints = [f for f in available if "cam" in f.lower() or "zed" in f.lower()]
        raise KeyError(
            f"Frame '{frame_name}' not in the robot model. "
            f"Camera-like frames: {hints or '(none found)'}. "
            f"Run with --list-frames to see all {len(available)} frames."
        )

    qpos = motion_manager.get_joint_pos()
    poses = motion_manager.fk(
        frame_names=[frame_name], qpos=qpos, update_robot_state=False
    )
    pose = poses[frame_name]
    matrix = np.asarray(pose.np if hasattr(pose, "np") else pose, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Unexpected FK pose shape {matrix.shape} for '{frame_name}'")
    return matrix


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(
    camera_frame: str = "head_camera_left_optical_frame",
    list_frames: bool = False,
    roi: tuple[int, int, int, int] | None = None,
    floor_roi: tuple[int, int, int, int] | None = None,
    link_frame_convention: bool = False,
    sensor: str = "head_camera",
    n_frames: int = 10,
    z_range: tuple[float, float] = (0.15, 5.0),
    ransac_threshold: float = 0.006,
    ransac_iterations: int = 400,
    seed: int = 0,
    show: bool = True,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> None:
    """Measure a planar surface's height in the robot base frame.

    Args:
        camera_frame: URDF frame name of the camera used for FK.
        list_frames: Print all model frame names and exit.
        roi: Target plane ROI as x0 y0 x1 y1; prompts interactively if omitted.
        floor_roi: Optional floor ROI used as a sanity check (should report a
            height near 0 and tilt near 0).
        link_frame_convention: Set when ``camera_frame`` is a ROS link frame
            (x forward, y left, z up) instead of an optical frame.
        sensor: Camera sensor name in the robot config.
        n_frames: Depth frames to median-filter together.
        z_range: Valid depth range in meters.
        ransac_threshold: Plane inlier threshold in meters.
        ransac_iterations: RANSAC hypothesis count.
        seed: RNG seed.
        show: Overlay the fitted inliers on the RGB image at the end.
        fx: Override focal length x (skips camera-info parsing).
        fy: Override focal length y.
        cx: Override principal point x.
        cy: Override principal point y.
    """
    configs = get_robot_config()

    if not configs.has_sensor(sensor):
        logger.error(f"'{sensor}' is not available on this robot configuration")
        return
    configs.enable_sensor(sensor)
    configs.sensors[sensor].transport = "zenoh"  # depth is Zenoh-only

    with Robot(configs=configs) as bot:
        motion_manager = build_motion_manager(bot)

        if list_frames:
            names = list_frame_names(motion_manager)
            print(f"{len(names)} frames in the robot model:")
            for name in names:
                marker = (
                    "  <-- camera?"
                    if ("cam" in name.lower() or "zed" in name.lower())
                    else ""
                )
                print(f"  {name}{marker}")
            return

        transform_base_cam = get_camera_to_base(motion_manager, camera_frame)
        if link_frame_convention:
            optical_to_link = np.eye(4)
            optical_to_link[:3, :3] = R_LINK_FROM_OPTICAL
            transform_base_cam = transform_base_cam @ optical_to_link

        logger.info(
            f"camera origin in base frame: "
            f"{np.round(transform_base_cam[:3, 3], 4).tolist()} m"
        )

        camera = getattr(bot.sensors, sensor)
        if not camera.wait_for_active(timeout=5.0):
            logger.warning("Camera streams did not all report active; continuing")

        rgb, depth = grab_frames(camera, n_frames=n_frames, timeout=15.0)
        logger.info(f"rgb {rgb.shape}, depth {depth.shape}")
        if rgb.shape[:2] != depth.shape[:2]:
            logger.warning(
                f"RGB {rgb.shape[:2]} and depth {depth.shape} differ in size; "
                "ROI pixels are interpreted in depth coordinates"
            )

        if None not in (fx, fy, cx, cy):
            intrinsics = (float(fx), float(fy), float(cx), float(cy))  # type: ignore[arg-type]
        else:
            info = camera.get_camera_info()
            if info is None:
                logger.error(
                    f"No response from 'sensors/{sensor}/info'. Pass "
                    "--fx/--fy/--cx/--cy manually."
                )
                return
            intrinsics = extract_intrinsics(info, depth.shape)
        logger.info(
            f"intrinsics fx={intrinsics[0]:.2f} fy={intrinsics[1]:.2f} "
            f"cx={intrinsics[2]:.2f} cy={intrinsics[3]:.2f}"
        )

        if roi is None:
            roi = select_roi(rgb, "Select the TARGET plane (e.g. the table top)")
            if roi is None:
                logger.error("No target ROI selected")
                return
        if floor_roi is None and show:
            floor_roi = select_roi(
                rgb, "Optional: select the FLOOR for a sanity check (q to skip)"
            )

        targets: list[tuple[str, tuple[int, int, int, int]]] = [("target", roi)]
        if floor_roi is not None:
            targets.append(("floor", floor_roi))

        results: dict[str, dict[str, Any]] = {}
        for label, box in targets:
            points_cam, pixels = back_project(depth, intrinsics, box, z_range)
            if len(points_cam) < 100:
                logger.error(
                    f"[{label}] only {len(points_cam)} valid depth pixels in "
                    f"ROI {box}; pick a larger region or widen --z-range"
                )
                continue
            points_base = transform_points(points_cam, transform_base_cam)
            result = analyze_plane(
                points_base,
                threshold=ransac_threshold,
                iterations=ransac_iterations,
                seed=seed,
            )
            result["pixels"] = pixels
            result["roi"] = box
            results[label] = result

        print("\n" + "=" * 66)
        print(f"PLANE HEIGHT IN BASE FRAME  (camera frame: {camera_frame})")
        print("=" * 66)
        for label, result in results.items():
            print(f"[{label}]  roi={result['roi']}")
            print(f"  height above base z=0 : {result['height_m'] * 1000:8.1f} mm")
            print(f"  tilt from base +z     : {result['tilt_deg']:8.2f} deg")
            print(f"  fit rms               : {result['rms_mm']:8.2f} mm")
            print(
                f"  inliers               : {result['n_inliers']:8d} / "
                f"{result['n_points']} ({result['inlier_ratio'] * 100:.1f}%)"
            )
            print(f"  normal (base)         : {np.round(result['normal'], 4).tolist()}")
            print(f"  centroid (base)       : {np.round(result['centroid'], 4).tolist()}")

            if result["inlier_ratio"] < 0.6 or result["rms_mm"] > 15.0:
                print(
                    "  WARNING: weak fit - the ROI probably contains more than "
                    "one surface. Do not trust this height."
                )

        if "floor" in results:
            floor = results["floor"]
            print("-" * 66)
            print(
                f"floor check: height={floor['height_m'] * 1000:+.1f} mm, "
                f"tilt={floor['tilt_deg']:.2f} deg "
                "(both should be near 0 if the frame and convention are right)"
            )
            if abs(floor["height_m"]) > 0.05 or floor["tilt_deg"] > 5.0:
                print(
                    "  WARNING: floor is off. Check --camera-frame and try "
                    "toggling --link-frame-convention."
                )
            if "target" in results:
                delta = results["target"]["height_m"] - floor["height_m"]
                print(f"target height above the measured floor: {delta * 1000:.1f} mm")
        print("=" * 66 + "\n")

        if show and "target" in results:
            target = results["target"]
            show_result(
                rgb,
                target["pixels"],
                target["inliers"],
                depth.shape,
                f"height = {target['height_m'] * 1000:.1f} mm "
                f"(tilt {target['tilt_deg']:.1f} deg, rms {target['rms_mm']:.1f} mm)",
            )


if __name__ == "__main__":
    tyro.cli(main)
