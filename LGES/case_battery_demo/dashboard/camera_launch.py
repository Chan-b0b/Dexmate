"""Best-effort remote launch of the nano's cameras over SSH.

The cameras are published by ``dexsensor`` running on the nano
(dexmate-nano@192.168.50.22), not on this machine. Without it, no frames flow
over zenoh and the dashboard image stays blank. This module lets the dashboard
bring that daemon up automatically instead of requiring a manual:

    ssh dexmate-nano@192.168.50.22
    dexsensor launch -s head_camera -s right_wrist_camera --config .../depth.toml

It relies on **key-based** SSH (BatchMode) — set it up once with
``ssh-copy-id dexmate-nano@192.168.50.22`` so no password is ever stored. Every
operation is best-effort and non-fatal: if the nano is unreachable the
dashboard still serves, the image is just blank until the camera appears.

The launch is idempotent (skips what is already Running) and detached
(``setsid``), so the daemon survives the SSH channel closing and is shared with
a running demo rather than tied to the dashboard's lifetime.

0917: the right-hand camera (ZED X One GS) was added to the default set. Two
things that follow from having more than one sensor:

  * "is it running" is asked of ``dexsensor list`` (the daemon's own view)
    instead of pattern-matching the launch command line. With several sensors
    the old pgrep pattern depended on the ORDER they were passed in, and a
    demo that launched the same sensors differently read as "not running".
  * if a dexsensor is already up but missing a sensor, the missing one is
    switched on at RUNTIME (``dexsensor enable``) rather than by killing and
    relaunching the process — killing it would blink the head camera off under
    whatever demo is using it.
"""

from __future__ import annotations

import re
import subprocess
import time

NANO_HOST = "dexmate-nano@192.168.50.22"
# The cameras the dashboard wants up. head_camera is the demo's eye; the hand
# camera is here so ./run_dashboard_demo.sh gives you both without a second step.
SENSORS: tuple[str, ...] = ("head_camera", "right_wrist_camera")
SENSOR = SENSORS[0]                  # kept for callers that named it explicitly
CONFIG = "/home/dexmate-nano/.dexmate/sensors/depth.toml"
REMOTE_LOG = "/tmp/dexsensor_head_camera.log"

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
             "-o", "StrictHostKeyChecking=accept-new"]


def _ssh(host: str, remote_cmd: str, timeout: float = 12.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *_SSH_OPTS, host, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def running_sensors(host: str = NANO_HOST) -> set[str]:
    """Sensor ids the nano's dexsensor currently reports as Running.

    ``dexsensor list`` is run through a LOGIN shell because it needs the nano's
    ROBOT_NAME / zenoh config to reach the daemon's control service; without
    them it prints the config's static states instead of the live ones.
    """
    try:
        r = _ssh(host, "bash -lc 'dexsensor list 2>/dev/null'", timeout=20.0)
    except (subprocess.TimeoutExpired, OSError):
        return set()
    return {m.group(1) for m in
            re.finditer(r"^(\S+)\s+\S+\s+Running\b", r.stdout, re.MULTILINE | re.IGNORECASE)}


def is_running(host: str = NANO_HOST, sensor: str = SENSOR, config: str = CONFIG) -> bool:
    """True if *sensor* is Running on *host*. (*config* is accepted and ignored:
    the daemon's own state is authoritative now — see the module docstring.)"""
    return sensor in running_sensors(host)


def _dexsensor_alive(host: str) -> bool:
    """Is there a dexsensor launch process at all? The bracket in "[d]exsensor"
    keeps pgrep from matching the wrapping shell, whose command line contains
    the pattern string itself."""
    try:
        r = _ssh(host, "pgrep -f '[d]exsensor launch' >/dev/null && echo YES || echo NO")
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "YES" in r.stdout


def ensure_camera(
    host: str = NANO_HOST,
    sensors: tuple[str, ...] | list[str] | None = None,
    config: str = CONFIG,
    verbose: bool = True,
    sensor: str | None = None,
) -> bool:
    """Make sure *sensors* are Running on *host*.

    Returns True if they all are (already, or after a successful start), False
    on any failure (unreachable host, a sensor that refused to initialise).
    Never raises. *sensor* is the old single-camera argument, still honoured.
    """
    wanted = tuple(sensors) if sensors else ((sensor,) if sensor else SENSORS)

    def say(msg: str) -> None:
        if verbose:
            print(f"[camera] {msg}")

    try:
        reachable = _ssh(host, "true").returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        reachable = False
    if not reachable:
        say(f"{host} unreachable over SSH (key auth set up? `ssh-copy-id {host}`). "
            "Dashboard will run; image stays blank until the camera is up.")
        return False

    have = running_sensors(host)
    if not have:
        # An empty read is ambiguous: it also happens when the daemon does not
        # answer that first control request in time, in which case dexsensor
        # falls back to printing the config's static states. Seen once on 0917,
        # and it costs a pointless enable of cameras that were already up, so
        # ask twice before concluding nothing is running.
        have = running_sensors(host)
    missing = [s for s in wanted if s not in have]
    if not missing:
        say(f"already running on {host}: {', '.join(wanted)} — nothing to do.")
        return True

    if _dexsensor_alive(host):
        # Something is already streaming (probably the head, under a demo).
        # Switch the rest on in place so that stream is never interrupted.
        say(f"dexsensor is up; enabling {', '.join(missing)} at runtime …")
        for s in missing:
            try:
                _ssh(host, f"bash -lc 'dexsensor enable {s}'", timeout=25.0)
            except (subprocess.TimeoutExpired, OSError) as e:
                say(f"enable {s} failed: {e}")
    else:
        say(f"launching {', '.join(wanted)} on {host} …")
        # Run through a LOGIN shell (bash -lc) so the nano's profile is sourced.
        # Over a plain non-interactive SSH command, ROBOT_NAME (and the zenoh
        # config) are unset, so dexsensor publishes under the wrong namespace
        # ("sensors/..." instead of "<ROBOT_NAME>/sensors/...") and subscribers
        # on the main computer never receive any frames.
        flags = " ".join(f"-s {s}" for s in wanted)
        launch = (
            f"setsid nohup bash -lc 'exec dexsensor launch {flags} --config {config}' "
            f"> {REMOTE_LOG} 2>&1 < /dev/null &"
        )
        try:
            _ssh(host, launch)
        except (subprocess.TimeoutExpired, OSError) as e:
            say(f"launch command failed: {e}")
            return False

    # Give the cameras time to come up, then confirm. The ZED cameras take a
    # few seconds each (GMSL open + calibration load), so this waits well past
    # the first poll before giving up.
    for _ in range(12):
        time.sleep(2.0)
        have = running_sensors(host)
        if all(s in have for s in wanted):
            say(f"running: {', '.join(wanted)} (logs: {host}:{REMOTE_LOG}).")
            return True
    still = [s for s in wanted if s not in have]
    say(f"did not come up: {', '.join(still)} — check {host}:{REMOTE_LOG}.")
    return False


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Bring the nano's cameras up over SSH.")
    p.add_argument("--host", default=NANO_HOST)
    p.add_argument("--sensors", nargs="+", default=list(SENSORS),
                   help=f"sensor ids to ensure (default: {' '.join(SENSORS)})")
    p.add_argument("--config", default=CONFIG)
    a = p.parse_args()
    ok = ensure_camera(host=a.host, sensors=tuple(a.sensors), config=a.config)
    sys.exit(0 if ok else 1)
