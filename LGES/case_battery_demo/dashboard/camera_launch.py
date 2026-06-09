"""Best-effort remote launch of the head-camera dexsensor over SSH.

The head camera is published by ``dexsensor`` running on the nano
(dexmate-nano@192.168.50.22), not on this machine. Without it, no frames flow
over zenoh and the dashboard image stays blank. This module lets the dashboard
bring that daemon up automatically instead of requiring a manual:

    ssh dexmate-nano@192.168.50.22
    dexsensor launch -s head_camera --config .../depth.toml

It relies on **key-based** SSH (BatchMode) — set it up once with
``ssh-copy-id dexmate-nano@192.168.50.22`` so no password is ever stored. Every
operation is best-effort and non-fatal: if the nano is unreachable the
dashboard still serves, the image is just blank until the camera appears.

The launch is idempotent (skips if already running) and detached (``setsid``),
so the daemon survives the SSH channel closing and is shared with a running
demo rather than tied to the dashboard's lifetime.
"""

from __future__ import annotations

import subprocess

NANO_HOST = "dexmate-nano@192.168.50.22"
SENSOR = "head_camera"
CONFIG = "/home/dexmate-nano/.dexmate/sensors/depth.toml"
REMOTE_LOG = "/tmp/dexsensor_head_camera.log"

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
             "-o", "StrictHostKeyChecking=accept-new"]


def _ssh(host: str, remote_cmd: str, timeout: float = 12.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *_SSH_OPTS, host, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def is_running(host: str = NANO_HOST, sensor: str = SENSOR) -> bool:
    """True if a dexsensor for *sensor* is already running on *host*.

    The pattern brackets the first char ("[d]exsensor") so pgrep matches the
    real dexsensor process but NOT the wrapping shell, whose command line
    literally contains the pattern string (a plain "dexsensor..." pattern
    self-matches and always reports running).
    """
    try:
        r = _ssh(host, f'pgrep -f "[d]exsensor.*{sensor}" >/dev/null && echo YES || echo NO')
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "YES" in r.stdout


def ensure_camera(
    host: str = NANO_HOST,
    sensor: str = SENSOR,
    config: str = CONFIG,
    verbose: bool = True,
) -> bool:
    """Make sure the head-camera dexsensor is running on *host*.

    Returns True if it is running (already, or after a successful launch),
    False on any failure (unreachable host, launch didn't take). Never raises.
    """
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

    if is_running(host, sensor):
        say(f"{sensor} already running on {host} — nothing to do.")
        return True

    say(f"launching {sensor} on {host} …")
    # Run through a LOGIN shell (bash -lc) so the nano's profile is sourced.
    # Over a plain non-interactive SSH command, ROBOT_NAME (and the zenoh
    # config) are unset, so dexsensor publishes under the wrong namespace
    # ("sensors/..." instead of "<ROBOT_NAME>/sensors/...") and subscribers on
    # the main computer never receive any frames.
    launch = (
        f"setsid nohup bash -lc 'exec dexsensor launch -s {sensor} --config {config}' "
        f"> {REMOTE_LOG} 2>&1 < /dev/null &"
    )
    try:
        _ssh(host, launch)
    except (subprocess.TimeoutExpired, OSError) as e:
        say(f"launch command failed: {e}")
        return False

    # Give dexsensor a moment to come up, then confirm.
    import time
    for _ in range(6):
        time.sleep(1.0)
        if is_running(host, sensor):
            say(f"{sensor} started (logs: {host}:{REMOTE_LOG}).")
            return True
    say(f"{sensor} did not appear after launch — check {host}:{REMOTE_LOG}.")
    return False


if __name__ == "__main__":
    import sys
    ok = ensure_camera()
    sys.exit(0 if ok else 1)
