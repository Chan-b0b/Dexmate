# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Robotiq 2F-85 gripper control over a USB-to-RS485 adapter.

Unlike robotiq_gripper_control.py (which routes Modbus RTU frames through the
robot arm's EE pass-through), this script talks to the gripper directly over a
serial port (e.g. /dev/ttyUSB0) using Modbus RTU at 115200 baud, 8N1.

Requirements:
    pip install pyserial

Usage:
    python robotiq_gripper_usb_control.py                    # auto-detect port
    python robotiq_gripper_usb_control.py --port /dev/ttyUSB0
    python robotiq_gripper_usb_control.py --close --close-strength 0.5
"""

import threading
import time
from typing import Annotated

import serial
import tyro
from loguru import logger
from serial.tools import list_ports

SLAVE_ID = 0x09
BAUD_RATE = 115200
STATUS_REQUEST_HEX = "09 04 07 D0 00 03 B1 CE"
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


def modbus_crc16(payload: bytes) -> int:
    """Compute Modbus RTU CRC-16 for the given payload."""
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


def append_crc(payload: bytes) -> bytes:
    """Append Modbus RTU CRC-16 (little-endian) to the payload."""
    crc = modbus_crc16(payload)
    return payload + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def build_write_registers_command(
    control_byte: int,
    target_position: int,
    speed: int = 0xFF,
    force: int = 0xFF,
) -> bytes:
    """Build Robotiq write-registers command with CRC."""
    payload = bytes(
        [
            SLAVE_ID,
            0x10,  # Function code: Write Multiple Registers
            0x03,
            0xE8,  # Start register
            0x00,
            0x03,  # Register count
            0x06,  # Byte count
            control_byte,
            0x00,
            0x00,
            target_position,
            speed,
            force,
        ]
    )
    return append_crc(payload)


def parse_status(payload: bytes) -> dict[str, int]:
    """Parse the 6-byte register payload of a status response."""
    return {
        "gACT": (payload[0] & 0b00000001),  # Gripper Activation Status
        "gGTO": (payload[0] & 0b00001000) >> 3,  # Action Status (go to)
        "gSTA": (payload[0] & 0b00110000) >> 4,  # Gripper Status
        "gOBJ": (payload[0] & 0b11000000) >> 6,  # Object Detection Status
        "gFLT": (payload[2] & 0b00001111),  # Fault Code
        "gPR": payload[3],  # 0 ~ 255, Position Request Echo
        "gPO": payload[4],  # 0 ~ 255, Actual Position
        "gCU": payload[5],  # 0 ~ 255, Motor Current
    }


class RobotiqGripperSerial:
    """Robotiq 2F-85 gripper over a USB-RS485 serial port (Modbus RTU)."""

    def __init__(self, port: str, timeout: float = 0.2) -> None:
        self._serial = serial.Serial(
            port=port,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        # One Modbus transaction (write request + read response) at a time;
        # the status monitor thread and main thread share this port.
        self._lock = threading.Lock()
        self._status_request = bytes.fromhex(STATUS_REQUEST_HEX.replace(" ", ""))

    def close(self) -> None:
        self._serial.close()

    def _transact(self, request: bytes, response_len: int) -> bytes | None:
        """Send a request and read a fixed-length response, verifying CRC."""
        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write(request)
            response = self._serial.read(response_len)

        if len(response) < response_len:
            logger.warning(
                f"Short/no response ({len(response)}/{response_len} bytes) "
                f"for request {request.hex(' ')}"
            )
            return None

        crc = modbus_crc16(response[:-2])
        if response[-2] != (crc & 0xFF) or response[-1] != ((crc >> 8) & 0xFF):
            logger.warning(f"CRC mismatch in response: {response.hex(' ')}")
            return None
        return response

    def write_control(
        self,
        control_byte: int,
        target_position: int = 0x00,
        speed: int = 0x00,
        force: int = 0x00,
    ) -> bool:
        """Write the gripper control registers. Returns True on valid echo."""
        request = build_write_registers_command(
            control_byte, target_position, speed, force
        )
        logger.info(f"Sending: {request.hex(' ')}")
        return self._transact(request, WRITE_RESPONSE_LEN) is not None

    def read_status(self) -> dict[str, int] | None:
        """Read and parse the gripper status registers."""
        response = self._transact(self._status_request, STATUS_RESPONSE_LEN)
        if response is None:
            return None
        if response[1] != 0x04 or response[2] != 0x06:
            logger.warning(f"Unexpected status response header: {response.hex(' ')}")
            return None
        return parse_status(response[3:-2])

    def activate(self, timeout: float = 10.0) -> bool:
        """Run the full activation sequence: reset (rACT=0) then activate (rACT=1).

        Waits until gSTA == 3 (activation complete). On success the gripper LED
        turns from red to blue.
        """
        logger.info("Resetting gripper (rACT=0)...")
        self.write_control(0x00)
        time.sleep(0.5)

        logger.info("Activating gripper (rACT=1)...")
        self.write_control(0x01)

        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.read_status()
            if status is not None:
                if status["gFLT"] != 0:
                    desc = FAULT_DESCRIPTIONS.get(status["gFLT"], "Unknown fault")
                    logger.error(
                        f"Gripper fault during activation: "
                        f"gFLT={status['gFLT']} (0x{status['gFLT']:02X}) — {desc}"
                    )
                    return False
                if status["gSTA"] == 3:
                    logger.info("Activation complete (gSTA=3). LED should be blue now.")
                    return True
            time.sleep(0.2)

        logger.error("Activation timed out")
        return False

    def move(self, target_position: int, speed: int = 0xFF, force: int = 0xFF) -> bool:
        """Command a move (rACT=1, rGTO=1) to target_position (0=open, 255=closed)."""
        return self.write_control(0x09, target_position, speed, force)


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
        logger.error(f"Multiple USB serial devices found: {candidates}. Use --port.")
    return None


def status_monitor_thread(gripper: RobotiqGripperSerial, stop_event: threading.Event):
    """Continuously poll and log gripper status."""
    while not stop_event.is_set():
        status = gripper.read_status()
        if status is not None:
            logger.info(f"gPO: {status['gPO']}, Status: {status}")
        time.sleep(0.5)


def main(
    port: str | None = None,
    open_cmd: Annotated[bool, tyro.conf.arg(name="open")] = False,
    close_cmd: Annotated[bool, tyro.conf.arg(name="close")] = False,
    close_strength: float = 1.0,
    action_interval: float = 5.0,
    monitor: bool = True,
) -> None:
    """Control a Robotiq 2F-85 gripper over a USB-RS485 adapter.

    Args:
        port: Serial port of the USB-RS485 adapter (e.g. /dev/ttyUSB0).
            Auto-detected if omitted and exactly one USB serial device exists.
        open_cmd: Send open command (CLI: --open).
        close_cmd: Send close command (CLI: --close).
        close_strength: Close target from 0.0 (open) to 1.0 (fully closed).
        action_interval: Delay between actions in seconds.
        monitor: Whether to run the background status polling thread.
    """
    if not 0.0 <= close_strength <= 1.0:
        raise ValueError(f"close_strength must be in [0.0, 1.0], got {close_strength}")

    if action_interval < 0.0:
        raise ValueError(f"action_interval must be >= 0.0, got {action_interval}")

    if not open_cmd and not close_cmd:
        logger.info("No --open/--close specified, defaulting to --close --open")
        close_cmd = True
        open_cmd = True

    if port is None:
        port = find_serial_port()
        if port is None:
            return
    logger.info(f"Using serial port: {port} @ {BAUD_RATE} baud")

    gripper = RobotiqGripperSerial(port)
    stop_event = threading.Event()
    monitor_thread = None

    try:
        if not gripper.activate():
            logger.error(
                "Gripper did not activate. Check RS485 wiring (A/B lines), "
                "gripper 24V power, and that no other program holds the port."
            )
            return

        if monitor:
            monitor_thread = threading.Thread(
                target=status_monitor_thread, args=(gripper, stop_event), daemon=True
            )
            monitor_thread.start()
            logger.info("Started status monitoring thread")

        if close_cmd:
            target_position = int(round(255 * close_strength))
            logger.info(
                f"Closing: strength={close_strength:.3f}, target_position={target_position}"
            )
            gripper.move(target_position)
            if open_cmd and action_interval > 0.0:
                time.sleep(action_interval)

        if open_cmd:
            logger.info("Opening gripper")
            gripper.move(0x00)
            if action_interval > 0.0:
                time.sleep(action_interval)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        if monitor_thread is not None:
            logger.info("Stopping status monitoring thread...")
            stop_event.set()
            monitor_thread.join(timeout=1.0)
        gripper.close()
        logger.info("Serial port closed")


if __name__ == "__main__":
    tyro.cli(main)
