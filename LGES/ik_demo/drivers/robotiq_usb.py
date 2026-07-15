"""Robotiq 2F gripper control over a USB-to-RS485 serial adapter (Modbus RTU).

Direct-serial replacement for drivers/robotiq.py (which routes Modbus frames
through the right arm's EE pass-through): same high-level interface
(initialize / open / partial_open / close / is_object_grasped) and config
knobs, but frames go straight out a serial port (e.g. /dev/ttyUSB0) at
115200 baud, 8N1. Adapted from robotiq_gripper_usb_control.py (repo root).

Modbus map (slave id from cfg.ROBOTIQ_SLAVE_ID):
    move   : FC 0x10 @ 0x03E8, 3 regs
    status : FC 0x04 @ 0x07D0, 3 regs
Position is 0 = open .. 255 = closed.
"""

from __future__ import annotations

import threading
import time

import serial
from loguru import logger
from serial.tools import list_ports

try:
    from .. import config as cfg
    from .robotiq import GripperStatus
except ImportError:  # allow running a module directly from ik_demo/
    import config as cfg
    from drivers.robotiq import GripperStatus

BAUD_RATE = 115200
# Response lengths: [slave, func, byte_count] + payload + 2-byte CRC
WRITE_RESPONSE_LEN = 8  # echo of write-multiple-registers request
STATUS_RESPONSE_LEN = 11  # 3 header + 6 data + 2 CRC

# gFLT fault codes from the Robotiq 2F-85/2F-140 instruction manual
FAULT_DESCRIPTIONS = {
    0x00: "No fault",
    0x05: "Action delayed; activation (re)activation must be completed first",
    0x07: "The activation bit must be set prior to action",
    0x08: "Maximum operating temperature exceeded",
    0x09: "No communication during at least 1 second",
    0x0A: "Under minimum operating voltage",
    0x0B: "Automatic release in progress",
    0x0C: "Internal fault",
    0x0D: "Activation fault (check that nothing interferes with the fingers)",
    0x0E: "Overcurrent protection triggered",
    0x0F: "Automatic release completed",
}


def find_serial_port() -> str | None:
    """Auto-detect a USB serial port; return None if ambiguous or absent."""
    candidates = [
        p.device
        for p in list_ports.comports()
        if "ttyUSB" in p.device or "ttyACM" in p.device
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        logger.error("No USB serial device found (/dev/ttyUSB* or /dev/ttyACM*)")
    else:
        logger.error(f"Multiple USB serial devices found: {candidates}. Set cfg.ROBOTIQ_USB_PORT.")
    return None


class RobotiqGripperUSB:
    """Robotiq 2F gripper over a USB-RS485 serial port (Modbus RTU)."""

    def __init__(self, port: str | None = None, timeout: float = 0.2) -> None:
        self._port = port if port is not None else cfg.ROBOTIQ_USB_PORT
        self._timeout = timeout
        self._serial: serial.Serial | None = None
        # One Modbus transaction (write request + read response) at a time.
        self._lock = threading.Lock()
        self._slave = cfg.ROBOTIQ_SLAVE_ID
        self._last_status: GripperStatus | None = None

    @property
    def available(self) -> bool:
        """True once the serial port is open."""
        return self._serial is not None and self._serial.is_open

    def connect(self) -> bool:
        """Open the serial port (auto-detected if cfg.ROBOTIQ_USB_PORT is None)."""
        if self.available:
            return True
        port = self._port or find_serial_port()
        if port is None:
            return False
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=BAUD_RATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._timeout,
            )
        except serial.SerialException as exc:
            logger.warning("[Robotiq-USB] cannot open {}: {}", port, exc)
            self._serial = None
            return False
        logger.info("[Robotiq-USB] connected on {} @ {} baud", port, BAUD_RATE)
        return True

    def disconnect(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

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

    def _transact(self, payload: bytes, response_len: int) -> bytes | None:
        """Send a framed request and read a fixed-length response, verifying CRC."""
        if not self.available:
            return None
        request = self._frame(payload)
        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write(request)
            response = self._serial.read(response_len)

        if len(response) < response_len:
            logger.warning(
                "[Robotiq-USB] short/no response ({}/{} bytes) for request {}",
                len(response), response_len, request.hex(" "),
            )
            return None
        crc = self._crc16(response[:-2])
        if response[-2] != (crc & 0xFF) or response[-1] != ((crc >> 8) & 0xFF):
            logger.warning("[Robotiq-USB] CRC mismatch in response: {}", response.hex(" "))
            return None
        return response

    def _write_control(
        self, control_byte: int, pos: int = 0, speed: int = 0, force: int = 0
    ) -> bool:
        """Write the gripper control registers. Returns True on a valid echo."""
        payload = bytes([self._slave, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                         control_byte, 0x00, 0x00, pos, speed, force])
        return self._transact(payload, WRITE_RESPONSE_LEN) is not None

    def read_status(self) -> GripperStatus | None:
        response = self._transact(
            bytes([self._slave, 0x04, 0x07, 0xD0, 0x00, 0x03]), STATUS_RESPONSE_LEN
        )
        if response is None or response[1] != 0x04 or response[2] != 0x06:
            return None
        p = response[3:9]
        status = GripperStatus(
            gACT=p[0] & 0x01,
            gGTO=(p[0] >> 3) & 0x01,
            gSTA=(p[0] >> 4) & 0x03,
            gOBJ=(p[0] >> 6) & 0x03,
            gFLT=p[2] & 0x0F,
            gPR=p[3],
            gPO=p[4],
            gCU=p[5],
        )
        self._last_status = status
        return status

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
            status = self.read_status()
            if status is None:
                done_count = 0
                time.sleep(poll_interval)
                continue
            got_any = True
            if status.has_fault:
                logger.warning("[Robotiq-USB] fault gFLT={}", status.gFLT)
            if status.motion_done:
                done_count += 1
                if done_count >= required_done_count:
                    return True
            else:
                done_count = 0
            time.sleep(poll_interval)
        if not got_any:
            logger.warning("[Robotiq-USB] no status feedback — fallback sleep {:.1f}s", fallback_sleep)
        else:
            logger.warning(
                "[Robotiq-USB] motion not confirmed in {:.1f}s — fallback sleep {:.1f}s",
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
        self._write_control(0x00)

    def initialize(self) -> bool:
        """Open the port, then reset + activate. Returns True if gripper is ready.

        Run once at startup. Waits until gSTA == 3 (activation complete; the
        gripper LED turns from red to blue).
        """
        if not self.connect():
            logger.warning("[Robotiq-USB] serial port unavailable — gripper disabled")
            return False
        logger.info("[Robotiq-USB] initializing (reset → activate)...")
        self.reset()
        time.sleep(0.5)
        self._write_control(0x01)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            status = self.read_status()
            if status is not None:
                if status.has_fault:
                    desc = FAULT_DESCRIPTIONS.get(status.gFLT, "Unknown fault")
                    logger.error(
                        "[Robotiq-USB] fault during activation: gFLT={} — {}",
                        status.gFLT, desc,
                    )
                    return False
                if status.gSTA == 3:
                    logger.info("[Robotiq-USB] activation complete")
                    return True
            time.sleep(0.2)
        logger.warning("[Robotiq-USB] activation did not complete")
        return False

    def goto(self, pos: int, speed: int | None = None, force: int | None = None) -> bool:
        """Move to raw position (0=open .. 255=closed) and wait for completion."""
        pos = max(0, min(255, int(pos)))
        speed = cfg.ROBOTIQ_SPEED if speed is None else max(0, min(255, speed))
        force = cfg.ROBOTIQ_FORCE if force is None else max(0, min(255, force))
        self._write_control(0x09, pos, speed, force)
        time.sleep(0.15)  # let command settle before polling status
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
                "[Robotiq-USB] gOBJ==3 but gap={} >= {} — treating as gripped",
                gap, cfg.ROBOTIQ_GRIP_MIN_GAP,
            )
            return True
        return False
