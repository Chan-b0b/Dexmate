"""Module 1 — Battery Detection.

Detects batteries in the source box using the ZED head camera (RGB + depth).
Returns a list of BatteryObservation objects in the robot base frame, sorted
nearest-first (topmost layer first).

Usage::

    from battery_pick.perception import BatteryDetector

    with Robot() as bot:
        detector = BatteryDetector(bot)
        observations = detector.detect()
        for obs in observations:
            print(obs.position_base)  # [x, y, z] in base frame (m)
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from loguru import logger

# Resolve sibling imports: LGES/ and LGES/../dex/
_LGES_DIR = os.path.join(os.path.dirname(__file__), "..")
_DEX_DIR = os.path.join(_LGES_DIR, "..", "dex")
for _p in (_LGES_DIR, _DEX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from box_detection_utils import transform_zed_point_to_base  # noqa: E402
import battery_pick.config as cfg  # noqa: E402


class BatteryObservation:
    """A single detected battery target.

    Attributes:
        position_base: [x, y, z] position in robot base frame (meters).
        pixel_center: (u, v) image coordinates for diagnostics / visualisation.
        depth_m: Raw depth reading at the battery centroid (meters).
    """

    __slots__ = ("position_base", "pixel_center", "depth_m")

    def __init__(
        self,
        position_base: np.ndarray,
        pixel_center: tuple[int, int],
        depth_m: float,
    ) -> None:
        self.position_base = position_base
        self.pixel_center = pixel_center
        self.depth_m = depth_m

    def __repr__(self) -> str:
        x, y, z = self.position_base
        return (
            f"BatteryObservation(base=({x:.3f}, {y:.3f}, {z:.3f})m, "
            f"depth={self.depth_m:.3f}m, px={self.pixel_center})"
        )


class BatteryDetector:
    """Detect batteries in the source box using ZED RGB+depth.

    Algorithm:
        1. Crop depth image to the configured source-box ROI.
        2. Find the minimum valid depth in the ROI — this is the top surface
           of the uppermost battery layer.
        3. Threshold to a ±BATTERY_TOP_DEPTH_TOL band around that minimum.
        4. Extract contours; filter by minimum area.
        5. Compute centroid of each contour, project to 3D using camera
           intrinsics and the robot FK (torso + head joints).
        6. Sort by depth ascending (nearest → topmost → pick first).

    Args:
        robot: Live Robot instance with ``head_camera`` sensor active.
    """

    def __init__(self, robot) -> None:
        self._robot = robot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> list[BatteryObservation]:
        """Capture one frame and return detected BatteryObservations.

        Returns:
            List sorted by depth ascending (topmost battery first).
            Empty list when camera data is unavailable or no batteries found.
        """
        camera_data = self._robot.sensors.head_camera.get_obs(
            obs_keys=["left_rgb", "depth"], include_timestamp=True
        )

        rgb_data = camera_data.get("left_rgb")
        depth_data = camera_data.get("depth")

        if rgb_data is None or depth_data is None:
            logger.warning("[Perception] Camera data unavailable")
            return []

        rgb_img = rgb_data["data"] if isinstance(rgb_data, dict) else rgb_data
        depth_img = depth_data["data"] if isinstance(depth_data, dict) else depth_data

        if rgb_img is None or depth_img is None:
            logger.warning("[Perception] Image data is None")
            return []

        return self._detect_from_images(rgb_img, depth_img)

    def detect_with_debug_image(
        self,
    ) -> tuple[list[BatteryObservation], np.ndarray | None]:
        """Same as detect() but also returns an annotated debug image.

        Returns:
            (observations, debug_bgr_image).  debug_bgr_image is None when
            camera data is unavailable.
        """
        camera_data = self._robot.sensors.head_camera.get_obs(
            obs_keys=["left_rgb", "depth"], include_timestamp=True
        )
        rgb_data = camera_data.get("left_rgb")
        depth_data = camera_data.get("depth")

        if rgb_data is None or depth_data is None:
            return [], None

        rgb_img = rgb_data["data"] if isinstance(rgb_data, dict) else rgb_data
        depth_img = depth_data["data"] if isinstance(depth_data, dict) else depth_data

        if rgb_img is None or depth_img is None:
            return [], None

        observations = self._detect_from_images(rgb_img, depth_img)
        debug_img = self._draw_detections(rgb_img, observations)
        return observations, debug_img

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_from_images(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
    ) -> list[BatteryObservation]:
        x1, y1, x2, y2 = cfg.SOURCE_BOX_ROI
        depth_roi = depth_img[y1:y2, x1:x2].copy().astype(np.float32)

        # Mask out invalid pixels
        valid = (
            np.isfinite(depth_roi)
            & (depth_roi > cfg.DEPTH_MIN_M)
            & (depth_roi < cfg.DEPTH_MAX_M)
        )
        if not valid.any():
            logger.warning("[Perception] No valid depth in source-box ROI")
            return []

        depth_clean = np.where(valid, depth_roi, np.float32(cfg.DEPTH_MAX_M))
        min_depth = float(depth_clean[valid].min())

        # Pixels belonging to the battery top surface
        top_mask = valid & (depth_clean - min_depth < cfg.BATTERY_TOP_DEPTH_TOL_M)
        binary = (top_mask.astype(np.uint8)) * 255

        # Morphological cleanup to merge nearby fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        q_torso = np.array(self._robot.torso.get_joint_pos(), dtype=np.float64)
        q_head = np.array(self._robot.head.get_joint_pos(), dtype=np.float64)

        observations: list[BatteryObservation] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.MIN_BATTERY_AREA_PX:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            # Centroid in ROI-local coords → full-image coords
            u_roi = int(M["m10"] / M["m00"])
            v_roi = int(M["m01"] / M["m00"])
            u = int(np.clip(u_roi + x1, 0, depth_img.shape[1] - 1))
            v = int(np.clip(v_roi + y1, 0, depth_img.shape[0] - 1))

            depth_val = float(depth_img[v, u])
            if not np.isfinite(depth_val) or not (cfg.DEPTH_MIN_M < depth_val < cfg.DEPTH_MAX_M):
                continue

            # Project pixel + depth → camera-frame 3-D point
            z_c = depth_val
            x_c = ((u - cfg.CAMERA_CX) / cfg.CAMERA_FX) * z_c
            y_c = ((v - cfg.CAMERA_CY) / cfg.CAMERA_FY) * z_c
            point_cam = np.array([x_c, y_c, z_c])

            point_base = transform_zed_point_to_base(point_cam, q_torso, q_head)

            observations.append(
                BatteryObservation(
                    position_base=point_base,
                    pixel_center=(u, v),
                    depth_m=depth_val,
                )
            )
            logger.debug(
                "[Perception] battery depth={:.3f}m  base=({:.3f}, {:.3f}, {:.3f})",
                depth_val,
                *point_base.tolist(),
            )

        # Sort nearest (topmost layer) first
        observations.sort(key=lambda o: o.depth_m)
        logger.info("[Perception] Detected {} batteries", len(observations))
        return observations

    @staticmethod
    def _draw_detections(
        rgb_img: np.ndarray,
        observations: list[BatteryObservation],
    ) -> np.ndarray:
        """Return a BGR copy of rgb_img with detections annotated."""
        debug = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) if rgb_img.shape[2] == 3 else rgb_img.copy()
        x1, y1, x2, y2 = cfg.SOURCE_BOX_ROI
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for i, obs in enumerate(observations):
            u, v = obs.pixel_center
            label = f"#{i} {obs.depth_m:.2f}m"
            cv2.circle(debug, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(debug, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return debug
