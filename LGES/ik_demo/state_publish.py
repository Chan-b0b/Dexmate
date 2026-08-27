"""Background publisher that mirrors live joint state onto zenoh topics.

Publishes each of {left_arm, right_arm, head, torso}'s pos/vel/torque as a
JointStateCodec-encoded dexcomm message, on the same topic convention the
robot's own state publishers use (dm/<robot>/state/arm/left, .../arm/right,
.../state/head, .../state/torso). Lets an external zenoh subscriber (e.g. an
Isaac Sim mirror) track the robot's joints without going through dexcontrol
itself.

Runs on a daemon thread inside the demo process, same lifecycle as
DashboardPublisher (start()/stop(), context manager).
"""

from __future__ import annotations

import threading
import time

from dexcomm import Publisher
from dexcomm.codecs import JointStateCodec
from dexcontrol.utils.os_utils import resolve_key_name
from loguru import logger

_COMPONENT_TOPICS = {
    "left_arm": "state/arm/left",
    "right_arm": "state/arm/right",
    "head": "state/head",
    "torso": "state/torso",
}


class StatePublisher:
    """Samples joint state on a daemon thread and publishes it over zenoh."""

    def __init__(self, robot, hz: float = 30.0) -> None:
        self._robot = robot
        self._period = 1.0 / max(hz, 1.0)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._publishers: dict[str, Publisher] = {}

    def start(self) -> "StatePublisher":
        if self._thread is not None:
            return self
        for comp, topic in _COMPONENT_TOPICS.items():
            if self._robot.has_component(comp):
                self._publishers[comp] = Publisher(
                    resolve_key_name(topic), encoder=JointStateCodec.encode
                )
            else:
                logger.warning("[state_publish] component {} unavailable — skipping", comp)
        self._thread = threading.Thread(
            target=self._run, name="state-publisher", daemon=True
        )
        self._thread.start()
        logger.info(
            "[state_publish] publishing joint state on: {}",
            ", ".join(resolve_key_name(t) for t in _COMPONENT_TOPICS.values()),
        )
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for pub in self._publishers.values():
            pub.shutdown()
        self._publishers.clear()

    def __enter__(self) -> "StatePublisher":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _run(self) -> None:
        fails = 0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._publish_once()
                fails = 0
            except Exception as e:  # noqa: BLE001 - never let one bad read kill the thread
                fails += 1
                if fails == 1 or fails % 100 == 0:
                    logger.warning("[state_publish] publish error x{}: {}", fails, e)
            dt = time.time() - t0
            self._stop.wait(max(0.0, self._period - dt))

    def _publish_once(self) -> None:
        ts = time.time_ns()
        for comp, pub in self._publishers.items():
            component = getattr(self._robot, comp)
            msg: dict = {"timestamp_ns": ts}
            try:
                msg["pos"] = component.get_joint_pos().tolist()
            except Exception:  # noqa: BLE001
                continue  # no position this tick — nothing worth publishing
            try:
                msg["vel"] = component.get_joint_vel().tolist()
            except Exception:  # noqa: BLE001
                pass
            try:
                msg["torque"] = component.get_joint_torque().tolist()
            except Exception:  # noqa: BLE001
                pass
            pub.publish(msg)
