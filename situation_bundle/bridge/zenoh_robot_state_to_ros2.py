# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Subscribe Vega Zenoh keys (dexcomm) and republish as ROS 2 Humble topics.

Designed for keys like ``dm/vg144e604acd-1p/state/arm/left`` — set ``--robot-prefix``
to the part before ``state/...`` (same idea as ``ROBOT_NAME`` in other examples).

Prerequisites:
  - ROS 2 Humble sourced (``rclpy``, ``sensor_msgs``, ``std_msgs``, ``geometry_msgs``)
  - ``dexcomm`` / ``dexcontrol`` on ``PYTHONPATH``
  - ``ZENOH_CONFIG`` if your deployment needs it

Example:
  export ZENOH_CONFIG=/path/to/zenoh_config.json5
  export ROBOT_PREFIX=dm/vg144e604acd-1p
  source /opt/ros/humble/setup.bash
  python zenoh_robot_state_to_ros2.py --ros-namespace /vega_1p
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import traceback
from queue import Empty, Full, Queue
from typing import Any, Callable

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from dexcomm import Node as DexcommNode
from dexcomm.codecs import (
    BMSStateCodec,
    DepthImageCodec,
    EStopStateCodec,
    IMUDataCodec,
    JointStateCodec,
    RGBImageCodec,
    UltrasonicStateCodec,
    WrenchStateCodec,
    WristButtonStateCodec,
)
from geometry_msgs.msg import Vector3, WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, Image, Imu, JointState
from std_msgs.msg import Bool, Float32MultiArray, String

try:
    from dexcomm.codecs import JsonDataCodec
except ImportError:
    JsonDataCodec = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

# Joint names aligned with examples/bridge/zenoh_to_ros2.py
JOINT_NAMES: dict[str, list[str]] = {
    "state/arm/left": [f"L_arm_j{i}" for i in range(1, 8)],
    "state/arm/right": [f"R_arm_j{i}" for i in range(1, 8)],
    "state/head": ["head_j1", "head_j2", "head_j3"],
    "state/torso": ["torso_j1", "torso_j2", "torso_j3"],
    "state/chassis/steer": ["L_wheel_j1", "R_wheel_j1"],
    "state/chassis/drive": ["L_wheel_j2", "R_wheel_j2"],
}


def _stamp(node: Node) -> Time:
    return node.get_clock().now().to_msg()


def _joint_state_from_dex(
    node: Node, names: list[str], data: dict[str, Any], frame_id: str = ""
) -> JointState:
    msg = JointState()
    msg.header.stamp = _stamp(node)
    msg.header.frame_id = frame_id
    msg.name = list(names)
    pos = data.get("pos", [])
    vel = data.get("vel", [])
    torque = data.get("torque", [])
    msg.position = [float(x) for x in pos]
    if vel is not None and len(vel) == len(names):
        msg.velocity = [float(x) for x in vel]
    if torque is not None and len(torque) == len(names):
        msg.effort = [float(x) for x in torque]
    return msg


def _battery_from_bms(node: Node, data: dict[str, Any]) -> BatteryState:
    msg = BatteryState()
    msg.header.stamp = _stamp(node)
    msg.voltage = float(data.get("voltage", 0.0))
    msg.temperature = float(data.get("temperature", float("nan")))
    msg.current = float(data.get("current", 0.0))
    msg.charge = float("nan")
    msg.capacity = float("nan")
    msg.design_capacity = float("nan")
    msg.percentage = float(data.get("percentage", 0.0)) / 100.0
    msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
    msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
    msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_UNKNOWN
    msg.present = True
    return msg


def _numpy_to_image_msg(
    node: Node,
    arr: np.ndarray,
    encoding: str,
    frame_id: str,
) -> Image:
    msg = Image()
    msg.header.stamp = _stamp(node)
    msg.header.frame_id = frame_id
    if arr.ndim == 2:
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    elif arr.ndim == 3:
        msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
    else:
        raise ValueError(f"Unsupported image ndim {arr.ndim}")
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding == "mono8":
        msg.step = msg.width
    elif encoding == "bgr8" or encoding == "rgb8":
        msg.step = msg.width * 3
    elif encoding == "16UC1":
        msg.step = msg.width * 2
    else:
        msg.step = int(arr.nbytes // msg.height) if msg.height else len(arr.tobytes())
    msg.data = arr.tobytes()
    return msg


def _image_from_decoded(node: Node, decoded: Any, frame_id: str) -> Image | None:
    """Accept ndarray (H,W) or (H,W,C) or dict with common keys."""
    if decoded is None:
        return None
    arr: np.ndarray | None = None
    enc = "bgr8"
    if isinstance(decoded, np.ndarray):
        arr = decoded
        if arr.ndim == 2:
            enc = "mono8" if arr.dtype == np.uint8 else "16UC1"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            enc = "bgr8"
    elif isinstance(decoded, dict):
        for k in ("image", "rgb", "bgr", "data", "frame"):
            if k in decoded and isinstance(decoded[k], np.ndarray):
                arr = decoded[k]
                break
        if arr is None:
            return None
    else:
        return None
    return _numpy_to_image_msg(node, np.ascontiguousarray(arr), enc, frame_id)


def _depth_from_decoded(node: Node, decoded: Any, frame_id: str) -> Image | None:
    if isinstance(decoded, np.ndarray):
        d = decoded
        if d.dtype != np.uint16:
            d = d.astype(np.uint16)
        return _numpy_to_image_msg(node, d, "16UC1", frame_id)
    if isinstance(decoded, dict):
        for k in ("depth", "image", "data"):
            if k in decoded and isinstance(decoded[k], np.ndarray):
                return _depth_from_decoded(node, decoded[k], frame_id)
    return None


def _imu_from_dex(node: Node, data: dict[str, Any], frame_id: str) -> Imu:
    msg = Imu()
    msg.header.stamp = _stamp(node)
    msg.header.frame_id = frame_id
    if "gyro" in data:
        g = np.asarray(data["gyro"], dtype=np.float64).ravel()
        if g.size >= 3:
            msg.angular_velocity = Vector3(x=float(g[0]), y=float(g[1]), z=float(g[2]))
    if "acc" in data:
        a = np.asarray(data["acc"], dtype=np.float64).ravel()
        if a.size >= 3:
            msg.linear_acceleration = Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2]))
    if "quat" in data:
        q = np.asarray(data["quat"], dtype=np.float64).ravel()
        if q.size >= 4:
            msg.orientation.w = float(q[0])
            msg.orientation.x = float(q[1])
            msg.orientation.y = float(q[2])
            msg.orientation.z = float(q[3])
    return msg


def _wrench_from_dex(node: Node, data: dict[str, Any], frame_id: str) -> WrenchStamped:
    w = np.asarray(data.get("wrench", np.zeros(6)), dtype=np.float64).ravel()
    if w.size < 6:
        w = np.pad(w, (0, 6 - len(w)))
    msg = WrenchStamped()
    msg.header.stamp = _stamp(node)
    msg.header.frame_id = frame_id
    msg.wrench.force = Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2]))
    msg.wrench.torque = Vector3(x=float(w[3]), y=float(w[4]), z=float(w[5]))
    return msg


class RobotStateRos2Bridge(Node):
    """Queues Zenoh-decoded messages and publishes on a ROS timer (thread-safe)."""

    def __init__(
        self,
        robot_prefix: str,
        ros_namespace: str,
        timer_period: float,
        qos_depth: int,
    ) -> None:
        ns = ros_namespace.strip()
        if ns and not ns.startswith("/"):
            ns = "/" + ns
        super().__init__("zenoh_robot_state_to_ros2", namespace=ns or None)

        self._robot_prefix = robot_prefix.strip().strip("/")
        self._qos = QoSProfile(depth=qos_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subs: list[Any] = []
        self._queues: dict[str, Queue[Any]] = {}
        self._bridge_publishers: dict[str, Any] = {}
        self._lock = threading.Lock()

        def make_queue(_k: str) -> Queue[Any]:
            q: Queue[Any] = Queue(maxsize=1)
            self._queues[_k] = q
            return q

        def arm_cb(rel: str, names: list[str]):
            def _cb(msg: dict[str, Any]) -> None:
                self._push(rel, ("joint", (names, msg)))

            return _cb

        # --- ROS publishers ---
        for rel, names in JOINT_NAMES.items():
            key = rel.replace("/", "_")
            self._bridge_publishers[rel] = self.create_publisher(JointState, f"{key}", self._qos)

        self._bridge_publishers["state/bms"] = self.create_publisher(BatteryState, "battery", self._qos)
        self._bridge_publishers["state/estop"] = self.create_publisher(String, "estop_json", self._qos)
        self._bridge_publishers["state/ultrasonic"] = self.create_publisher(
            Float32MultiArray, "ultrasonic_ranges", self._qos
        )
        for side in ("left", "right"):
            wr = f"state/wrench/{side}"
            self._bridge_publishers[wr] = self.create_publisher(WrenchStamped, f"wrench_{side}", self._qos)
            wb = f"state/wrist_button/{side}"
            self._bridge_publishers[f"{wb}/blue"] = self.create_publisher(Bool, f"wrist_{side}_blue", self._qos)
            self._bridge_publishers[f"{wb}/green"] = self.create_publisher(Bool, f"wrist_{side}_green", self._qos)

        cam_base = "sensors/head_camera"
        self._bridge_publishers[f"{cam_base}/left_rgb"] = self.create_publisher(
            Image, "head_camera/left_rgb", self._qos
        )
        self._bridge_publishers[f"{cam_base}/right_rgb"] = self.create_publisher(
            Image, "head_camera/right_rgb", self._qos
        )
        self._bridge_publishers[f"{cam_base}/depth"] = self.create_publisher(
            Image, "head_camera/depth", self._qos
        )
        self._bridge_publishers[f"{cam_base}/imu"] = self.create_publisher(Imu, "head_camera/imu", self._qos)

        self._bridge_publishers["heartbeat"] = self.create_publisher(String, "heartbeat_json", self._qos)

        # --- Dexcomm node (namespace = robot prefix) ---
        self._dex = DexcommNode(name="ros2_telemetry_bridge", namespace=self._robot_prefix)

        def sub(rel: str, decoder: Callable[[bytes], Any], cb: Callable[[Any], None]) -> None:
            make_queue(rel)
            s = self._dex.create_subscriber(topic=rel, callback=cb, decoder=decoder)
            self._subs.append(s)

        for rel, names in JOINT_NAMES.items():
            sub(rel, JointStateCodec.decode, arm_cb(rel, names))

        sub("state/bms", BMSStateCodec.decode, lambda m: self._push("state/bms", ("bms", m)))
        sub("state/estop", EStopStateCodec.decode, lambda m: self._push("state/estop", ("estop", m)))
        sub(
            "state/ultrasonic",
            UltrasonicStateCodec.decode,
            lambda m: self._push("state/ultrasonic", ("ultra", m)),
        )
        sub(
            "state/wrench/left",
            WrenchStateCodec.decode,
            lambda m: self._push("state/wrench/left", ("wrench", m)),
        )
        sub(
            "state/wrench/right",
            WrenchStateCodec.decode,
            lambda m: self._push("state/wrench/right", ("wrench", m)),
        )
        sub(
            "state/wrist_button/left",
            WristButtonStateCodec.decode,
            lambda m: self._push("state/wrist_button/left", ("wbtn", m)),
        )
        sub(
            "state/wrist_button/right",
            WristButtonStateCodec.decode,
            lambda m: self._push("state/wrist_button/right", ("wbtn", m)),
        )

        sub(
            f"{cam_base}/left_rgb",
            RGBImageCodec.decode,
            lambda m: self._push(f"{cam_base}/left_rgb", ("rgb", m)),
        )
        sub(
            f"{cam_base}/right_rgb",
            RGBImageCodec.decode,
            lambda m: self._push(f"{cam_base}/right_rgb", ("rgb", m)),
        )
        sub(
            f"{cam_base}/depth",
            DepthImageCodec.decode,
            lambda m: self._push(f"{cam_base}/depth", ("depth", m)),
        )
        sub(
            f"{cam_base}/imu",
            IMUDataCodec.decode,
            lambda m: self._push(f"{cam_base}/imu", ("imu", m)),
        )

        if JsonDataCodec is not None:
            sub("heartbeat", JsonDataCodec.decode, lambda m: self._push("heartbeat", ("hb", m)))
        else:
            self.get_logger().warn("JsonDataCodec unavailable; skipping /heartbeat subscription")

        self.create_timer(timer_period, self._flush_all)
        self.get_logger().info(
            f"Zenoh prefix '{self._robot_prefix}' → ROS namespace '{ns or '/'}' "
            f"(timer {timer_period * 1000:.1f} ms)"
        )

    def _push(self, key: str, item: Any) -> None:
        q = self._queues.get(key)
        if q is None:
            return
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(item)
            except Full:
                pass

    def _flush_all(self) -> None:
        for key, q in list(self._queues.items()):
            try:
                kind, payload = q.get_nowait()
            except Empty:
                continue
            try:
                self._dispatch(key, str(kind), payload)
            except Exception as e:
                self.get_logger().error(f"Publish {key}: {e}\n{traceback.format_exc()}")

    def _dispatch(self, key: str, kind: str, payload: Any) -> None:
        if kind == "joint":
            names, data = payload[0], payload[1]
            pub = self._bridge_publishers.get(key)
            if pub:
                pub.publish(_joint_state_from_dex(self, names, data))
        elif kind == "bms":
            self._bridge_publishers["state/bms"].publish(_battery_from_bms(self, payload))
        elif kind == "estop":
            s = String()
            s.data = json.dumps(payload, default=str)
            self._bridge_publishers["state/estop"].publish(s)
        elif kind == "ultra":
            msg = Float32MultiArray()
            msg.data = [
                float(payload["front_left"]),
                float(payload["front_right"]),
                float(payload["back_left"]),
                float(payload["back_right"]),
            ]
            self._bridge_publishers["state/ultrasonic"].publish(msg)
        elif kind == "wrench":
            fid = "left_tool" if "left" in key else "right_tool"
            self._bridge_publishers[key].publish(_wrench_from_dex(self, payload, fid))
        elif kind == "wbtn":
            side = "left" if "left" in key else "right"
            b = Bool(data=bool(payload.get("blue_button", False)))
            g = Bool(data=bool(payload.get("green_button", False)))
            self._bridge_publishers[f"state/wrist_button/{side}/blue"].publish(b)
            self._bridge_publishers[f"state/wrist_button/{side}/green"].publish(g)
        elif kind == "rgb":
            img = _image_from_decoded(self, payload, "head_camera_rgb")
            if img is not None:
                self._bridge_publishers[key].publish(img)
        elif kind == "depth":
            img = _depth_from_decoded(self, payload, "head_camera_depth")
            if img is not None:
                self._bridge_publishers[key].publish(img)
        elif kind == "imu":
            self._bridge_publishers[key].publish(_imu_from_dex(self, payload, "head_camera_imu"))
        elif kind == "hb":
            s = String()
            s.data = json.dumps(payload, default=str)
            self._bridge_publishers["heartbeat"].publish(s)

    def shutdown_bridge(self) -> None:
        for s in self._subs:
            try:
                s.shutdown()
            except Exception as e:
                logger.warning("subscriber shutdown: %s", e)
        try:
            dex = self._dex
            shutdown = getattr(dex, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                close = getattr(dex, "close", None)
                if callable(close):
                    close()
        except Exception as e:
            logger.warning("dexcomm node teardown: %s", e)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-prefix",
        default=None,
        help="Zenoh key prefix without trailing slash, e.g. dm/vg144e604acd-1p (default: env ROBOT_PREFIX)",
    )
    parser.add_argument(
        "--ros-namespace",
        default="/vega",
        help="ROS 2 namespace for published topics (leading slash optional)",
    )
    parser.add_argument(
        "--timer-ms",
        type=float,
        default=10.0,
        help="ROS timer period in ms to drain Zenoh queues",
    )
    parser.add_argument(
        "--qos-depth",
        type=int,
        default=5,
        help="Publisher QoS queue depth",
    )
    args, ros_argv = parser.parse_known_args()

    import os

    robot_prefix = args.robot_prefix or os.environ.get("ROBOT_PREFIX", "").strip()
    if not robot_prefix:
        print("Set --robot-prefix or ROBOT_PREFIX (e.g. dm/vg144e604acd-1p)", file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=ros_argv)
    node: RobotStateRos2Bridge | None = None
    try:
        node = RobotStateRos2Bridge(
            robot_prefix=robot_prefix,
            ros_namespace=args.ros_namespace,
            timer_period=max(0.001, args.timer_ms / 1000.0),
            qos_depth=max(1, args.qos_depth),
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown_bridge()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
