"""Robotiq 2-finger gripper control over the right arm's EE pass-through.

The gripper hangs off the right arm and is reached as a raw RS485/Modbus-RTU
device through ``right_arm.send_ee_pass_through_message`` /
``get_ee_pass_through_response`` (available only while the EE type is UNKNOWN,
i.e. ``right_arm.enable_ee_pass_through is True``).

Modbus map (slave id from cfg.ROBOTIQ_SLAVE_ID):
    move   : FC 0x10 @ 0x03E8, 3 regs
    status : FC 0x04 @ 0x07D0, 3 regs
Position is 0 = open .. 255 = closed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from . import config as cfg


@dataclass
class GripperStatus:
    gACT: int
    gGTO: int
    gSTA: int
    gOBJ: int
    gFLT: int
    gPR: int
    gPO: int
    gCU: int

    @property
    def motion_done(self) -> bool:
        """gOBJ != 0: fingers stopped (object or target reached)."""
        return self.gOBJ in (1, 2, 3)

    @property
    def has_fault(self) -> bool:
        return self.gFLT != 0


# Last commanded finger position (0=open .. 255=closed), tracked at module
# level so the recorder can read it without an extra Modbus round-trip.
_last_cmd_pos: int | None = None


def commanded_pos() -> int | None:
    return _last_cmd_pos


class RobotiqGripper:
    """Robotiq 2F gripper via Dexmate EE pass-through (right arm, RS485/Modbus-RTU)."""

    def __init__(self, robot, side: str = "right") -> None:
        self._robot = robot
        self._side = side
        self._slave = cfg.ROBOTIQ_SLAVE_ID
        self._last_status: GripperStatus | None = None

    @property
    def _arm(self):
        return getattr(self._robot, f"{self._side}_arm")

    @property
    def available(self) -> bool:
        """True if the arm exposes the EE pass-through (UNKNOWN end effector)."""
        return bool(getattr(self._arm, "enable_ee_pass_through", False))

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _crc16(self, data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return crc

    def _frame(self, payload: bytes) -> bytes:
        crc = self._crc16(payload)
        return payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def _send(self, payload: bytes) -> None:
        self._arm.send_ee_pass_through_message(self._frame(payload))

    def _flush_responses(self, duration: float = 0.2) -> None:
        """Drain stale EE pass-through responses before a fresh status read."""
        start = time.time()
        while time.time() - start < duration:
            r = self._arm.get_ee_pass_through_response()
            if r is None:
                time.sleep(0.01)
            else:
                logger.debug("[Robotiq] flushed stale response: {}", r)

    def parse_status(self, data: bytes) -> GripperStatus | None:
        if len(data) < 9 or data[1] != 0x04 or data[2] != 0x06:
            return None
        p = data[3:9]
        return GripperStatus(
            gACT=p[0] & 0x01,
            gGTO=(p[0] >> 3) & 0x01,
            gSTA=(p[0] >> 4) & 0x03,
            gOBJ=(p[0] >> 6) & 0x03,
            gFLT=p[2] & 0x0F,
            gPR=p[3],
            gPO=p[4],
            gCU=p[5],
        )

    def read_status(self, timeout: float = 0.5) -> GripperStatus | None:
        self._send(bytes([self._slave, 0x04, 0x07, 0xD0, 0x00, 0x03]))
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self._arm.get_ee_pass_through_response()
            if r and r.get("data"):
                status = self.parse_status(bytes(r["data"]))
                if status is not None:
                    self._last_status = status
                    return status
            time.sleep(0.01)
        return None

    def wait_until_done(
        self,
        timeout: float = 3.0,
        poll_interval: float = 0.1,
        fallback_sleep: float = 1.5,
        required_done_count: int = 2,
    ) -> bool:
        """Block until gOBJ confirms motion done, with fallback sleep on timeout."""
        start = time.time()
        got_any = False
        done_count = 0
        while time.time() - start < timeout:
            status = self.read_status(timeout=0.3)
            if status is None:
                done_count = 0
                time.sleep(poll_interval)
                continue
            got_any = True
            if status.has_fault:
                logger.warning("[Robotiq] fault gFLT={}", status.gFLT)
            if status.motion_done:
                done_count += 1
                if done_count >= required_done_count:
                    return True
            else:
                done_count = 0
            time.sleep(poll_interval)
        if not got_any:
            logger.warning("[Robotiq] no status feedback — fallback sleep {:.1f}s", fallback_sleep)
        else:
            logger.warning(
                "[Robotiq] motion not confirmed in {:.1f}s — fallback sleep {:.1f}s",
                timeout, fallback_sleep,
            )
        if fallback_sleep > 0:
            time.sleep(fallback_sleep)
        return False

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset activation bit (clears fault state before activate)."""
        self._send(bytes([self._slave, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))

    def activate(self, wait: bool = True, timeout: float = 5.0) -> bool:
        """Activate the gripper (no-op if already activated). Returns True if ready."""
        if not self.available:
            logger.warning("[Robotiq] EE pass-through not available on {} arm", self._side)
            return False
        status = self.read_status()
        if status and status.gACT == 1 and status.gSTA == 3:
            return True
        logger.info("[Robotiq] activating...")
        self._send(bytes([self._slave, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                          0x01, 0x00, 0x00, 0x00, 0x00, 0x00]))
        if not wait:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.read_status()
            if status and status.gSTA == 3:
                logger.info("[Robotiq] activation complete")
                return True
            time.sleep(0.1)
        logger.warning("[Robotiq] activation did not complete within {:.1f}s", timeout)
        return False

    def initialize(self) -> bool:
        """Reset + activate. Returns True if gripper is ready.

        Run once at startup. reset() clears any pre-existing fault or partial
        activation state before activate() is sent.
        """
        if not self.available:
            logger.warning("[Robotiq] EE pass-through not available on {} arm", self._side)
            return False
        logger.info("[Robotiq] initializing (reset → activate)...")
        self.reset()
        time.sleep(0.5)
        return self.activate(wait=True)

    def goto(self, pos: int, speed: int | None = None, force: int | None = None) -> bool:
        """Move to raw position (0=open .. 255=closed) and wait for completion."""
        pos = max(0, min(255, int(pos)))
        speed = cfg.ROBOTIQ_SPEED if speed is None else max(0, min(255, speed))
        force = cfg.ROBOTIQ_FORCE if force is None else max(0, min(255, force))
        self._flush_responses(0.1)
        self._send(bytes([self._slave, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                          0x09, 0x00, 0x00, pos, speed, force]))
        global _last_cmd_pos
        _last_cmd_pos = pos
        time.sleep(0.15)  # let command settle before polling status
        self._flush_responses(0.1)
        return self.wait_until_done()

    def open(self) -> bool:
        return self.goto(cfg.ROBOTIQ_OPEN_POS)

    def partial_open(self) -> bool:
        """Open to ROBOTIQ_PARTIAL_OPEN_POS (avoids floor contact near the ground)."""
        return self.goto(cfg.ROBOTIQ_PARTIAL_OPEN_POS)

    def close(self) -> bool:
        return self.goto(cfg.ROBOTIQ_CLOSE_POS)

    def is_object_grasped(self) -> bool:
        """True if an object is held after close.

        gOBJ==2 is the primary signal. Slim-object fallback: gOBJ==3 but
        gPO stopped at least ROBOTIQ_GRIP_MIN_GAP counts short of CLOSE_POS.
        """
        status = self.read_status()
        if not status:
            return False
        if status.gOBJ == 2:
            return True
        gap = cfg.ROBOTIQ_CLOSE_POS - status.gPO
        if status.gOBJ == 3 and gap >= cfg.ROBOTIQ_GRIP_MIN_GAP:
            logger.debug(
                "[Robotiq] gOBJ==3 but gap={} >= {} — treating as gripped",
                gap, cfg.ROBOTIQ_GRIP_MIN_GAP,
            )
            return True
        return False
