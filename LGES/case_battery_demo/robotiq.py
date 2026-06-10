"""Robotiq 2-finger gripper control over the right arm's EE pass-through.

The gripper hangs off the right arm and is reached as a raw RS485/Modbus-RTU
device through ``right_arm.send_ee_pass_through_message`` /
``get_ee_pass_through_response`` (available only while the EE type is UNKNOWN,
i.e. ``right_arm.enable_ee_pass_through is True``).

Modbus map (slave id from cfg.ROBOTIQ_SLAVE_ID):
    move   : FC 0x10 @ 0x03E8, 3 regs -> 09 10 03 E8 00 03 06 09 00 00 <pos> <speed> <force>
    status : FC 0x04 @ 0x07D0, 3 regs -> body [status, _, fault, posEcho, position, current]
Position is 0 = open .. 255 = closed.
"""

from __future__ import annotations

import time

from loguru import logger

from . import config as cfg


def _crc16(data: bytes) -> bytes:
    """Modbus RTU CRC16, returned low-byte-first (as appended on the wire)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def _frame(payload: bytes) -> bytes:
    return payload + _crc16(payload)


# Last commanded finger position (0=open .. 255=closed), or None until first
# move. Tracked at module level (there is a single gripper in the demo) so the
# recorder can log the commanded gripper state per frame without a slow Modbus
# status read, mirroring suction_io.is_suction_commanded_on().
_last_cmd_pos: int | None = None


def commanded_pos() -> int | None:
    return _last_cmd_pos


class RobotiqGripper:
    """Minimal Robotiq control via EE pass-through on one arm (default right)."""

    def __init__(self, robot, side: str = "right") -> None:
        self._robot = robot
        self._side = side
        self._slave = cfg.ROBOTIQ_SLAVE_ID

    @property
    def _arm(self):
        return getattr(self._robot, f"{self._side}_arm")

    @property
    def available(self) -> bool:
        """True if the arm exposes the EE pass-through (UNKNOWN end effector)."""
        return bool(getattr(self._arm, "enable_ee_pass_through", False))

    # ------------------------------------------------------------------
    # Low-level send / receive
    # ------------------------------------------------------------------

    def _send(self, payload: bytes) -> None:
        self._arm.send_ee_pass_through_message(_frame(payload))

    def read_status(self, timeout: float = 0.5) -> dict | None:
        """Send a status read (FC04) and return the parsed gripper status.

        Returns a dict with gACT/gGTO/gSTA/gOBJ/gFLT/gPR/gPO/gCU, or None if no
        fresh FC04 reply arrived within *timeout*.
        """
        self._send(bytes([self._slave, 0x04, 0x07, 0xD0, 0x00, 0x03]))
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self._arm.get_ee_pass_through_response()
            if r and r.get("data"):
                d = bytes(r["data"])
                if len(d) >= 9 and d[0] == self._slave and d[1] == 0x04:
                    status, _reserved, fault, pos_echo, position, current = d[3:9]
                    return {
                        "gACT": status & 0x01,
                        "gGTO": (status >> 3) & 0x01,
                        "gSTA": (status >> 4) & 0x03,
                        "gOBJ": (status >> 6) & 0x03,
                        "gFLT": fault,
                        "gPR": pos_echo,
                        "gPO": position,
                        "gCU": current,
                    }
            time.sleep(0.01)
        return None

    def is_object_grasped(self) -> bool:
        """True if the gripper is holding an object after a close command.

        Primary signal: gOBJ == 2 (fingers stopped while closing on an object).
        Fallback for slim objects: gOBJ == 3 (reached requested position) but
        gPO stopped at least ROBOTIQ_GRIP_MIN_GAP counts short of CLOSE_POS —
        the object prevented full closure even though gOBJ didn't fire.
        """
        status = self.read_status()
        if not status:
            return False
        if status["gOBJ"] == 2:
            return True
        gap = cfg.ROBOTIQ_CLOSE_POS - status["gPO"]
        if status["gOBJ"] == 3 and gap >= cfg.ROBOTIQ_GRIP_MIN_GAP:
            logger.debug(
                "[Robotiq] gOBJ==3 but gap={} >= {} — treating as gripped",
                gap, cfg.ROBOTIQ_GRIP_MIN_GAP,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def activate(self, wait: bool = True, timeout: float = 5.0) -> bool:
        """Activate the gripper (no-op if already activated). Returns True if ready."""
        if not self.available:
            logger.warning("[Robotiq] EE pass-through not available on {} arm", self._side)
            return False
        status = self.read_status()
        if status and status["gACT"] == 1 and status["gSTA"] == 3:
            return True
        logger.info("[Robotiq] activating gripper...")
        self._send(bytes([self._slave, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                          0x01, 0x00, 0x00, 0x00, 0x00, 0x00]))
        if not wait:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.read_status()
            if status and status["gSTA"] == 3:
                logger.info("[Robotiq] activation complete")
                return True
            time.sleep(0.1)
        logger.warning("[Robotiq] activation did not complete within {:.1f}s", timeout)
        return False

    def goto(
        self,
        pos: int,
        speed: int | None = None,
        force: int | None = None,
        wait: bool = True,
        timeout: float = 4.0,
    ) -> int | None:
        """Command an absolute finger position (0=open .. 255=closed).

        If *wait*, blocks until the fingers stop (gOBJ != 0) or *timeout*, and
        returns the final position (gPO); otherwise returns None immediately.
        """
        pos = max(0, min(255, int(pos)))
        speed = cfg.ROBOTIQ_SPEED if speed is None else speed
        force = cfg.ROBOTIQ_FORCE if force is None else force
        self._send(bytes([self._slave, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                          0x09, 0x00, 0x00, pos, speed & 0xFF, force & 0xFF]))
        global _last_cmd_pos
        _last_cmd_pos = pos
        if not wait:
            return None
        time.sleep(0.1)
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            status = self.read_status()
            if status:
                last = status["gPO"]
                if status["gOBJ"] != 0:  # 1/2 = object, 3 = at requested position
                    return last
            time.sleep(0.05)
        return last

    def open(self, **kwargs) -> int | None:
        return self.goto(cfg.ROBOTIQ_OPEN_POS, **kwargs)

    def partial_open(self, **kwargs) -> int | None:
        """Open to ROBOTIQ_PARTIAL_OPEN_POS instead of fully open.

        Use when the gripper is near the ground and a full open would cause
        the lower finger to contact the floor.
        """
        return self.goto(cfg.ROBOTIQ_PARTIAL_OPEN_POS, **kwargs)

    def close(self, **kwargs) -> int | None:
        return self.goto(cfg.ROBOTIQ_CLOSE_POS, **kwargs)
