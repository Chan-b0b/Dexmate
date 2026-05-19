# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai
#

"""Launch optional Zenoh→ROS2 bridge + manipulation manager (Task Manager forwarding).

- ``bridge/zenoh_robot_state_to_ros2.py`` — if ``ROBOT_PREFIX`` is set and bridge not skipped.
- ``manipulation_bridge/manipulation_manager.py`` — unless ``SITUATION_SKIP_MANIPULATION_BRIDGE=1``.

``dexcomm`` is required only when the bridge process runs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# situation_bundle/situation_stack/main.py → bundle root
_BUNDLE_ROOT = Path(__file__).resolve().parent.parent


def _prepend_pythonpath(env: dict[str, str], extra: str) -> None:
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extra}{os.pathsep}{prev}" if prev else extra


def _check_imports_dexcomm() -> None:
    try:
        import dexcomm  # noqa: F401
    except ImportError as e:
        print(
            "dexcomm import failed. Install dexcomm/dexcontrol (pip) or set PYTHONPATH.",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def _bridge_enabled(env: dict[str, str]) -> bool:
    if os.environ.get("SITUATION_SKIP_ZENOH_ROS_BRIDGE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    return bool(env.get("ROBOT_PREFIX", "").strip())


def _maybe_bridge(env: dict[str, str]) -> tuple[str, list[str]] | None:
    if not _bridge_enabled(env):
        if not os.environ.get("SITUATION_SKIP_ZENOH_ROS_BRIDGE", "").lower() in (
            "1",
            "true",
            "yes",
        ) and not env.get("ROBOT_PREFIX", "").strip():
            print(
                "[situation_stack] ROBOT_PREFIX unset → skip zenoh_robot_state_to_ros2 "
                "(set ROBOT_PREFIX or SITUATION_SKIP_ZENOH_ROS_BRIDGE=1)",
                file=sys.stderr,
            )
        return None
    py = sys.executable
    script = str(_BUNDLE_ROOT / "bridge" / "zenoh_robot_state_to_ros2.py")
    ns = env.get("SITUATION_ROS_NAMESPACE", "/vega_1p")
    return ("zenoh_ros2_bridge", [py, script, "--ros-namespace", ns])


def _maybe_manipulation_manager(env: dict[str, str]) -> tuple[str, list[str]] | None:
    if env.get("SITUATION_SKIP_MANIPULATION_BRIDGE", "").lower() in ("1", "true", "yes"):
        return None
    py = sys.executable
    return (
        "manipulation_manager",
        [py, str(_BUNDLE_ROOT / "manipulation_bridge" / "manipulation_manager.py")],
    )


def _build_procs(env: dict[str, str]) -> list[tuple[str, subprocess.Popen[str]]]:
    children: list[tuple[str, list[str]]] = []
    br = _maybe_bridge(env)
    if br:
        children.append(br)
    mm = _maybe_manipulation_manager(env)
    if mm:
        children.append(mm)
    if not children:
        print(
            "[situation_stack] Nothing to run: enable the bridge (ROBOT_PREFIX) and/or "
            "manipulation manager (clear SITUATION_SKIP_MANIPULATION_BRIDGE), "
            "or clear SITUATION_SKIP_ZENOH_ROS_BRIDGE for bridge.",
            file=sys.stderr,
        )
        return []
    out: list[tuple[str, subprocess.Popen[str]]] = []
    for name, cmd in children:
        p = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(_BUNDLE_ROOT),
        )
        out.append((name, p))
        time.sleep(0.25)
    return out


def main() -> None:
    env = os.environ.copy()
    _prepend_pythonpath(env, str(_BUNDLE_ROOT))

    if _bridge_enabled(env):
        _check_imports_dexcomm()

    procs = _build_procs(env)
    if not procs:
        raise SystemExit(1)

    print(
        "[situation_stack] started: "
        + ", ".join(n for n, _ in procs)
        + " (Ctrl+C to stop)",
        file=sys.stderr,
    )

    def _stop_children(sig: int) -> None:
        for _name, p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(sig)
                except ProcessLookupError:
                    pass

    exit_code = 0
    try:
        while True:
            for name, p in procs:
                code = p.poll()
                if code is None:
                    continue
                print(f"[situation_stack] {name} exited with {code}", file=sys.stderr)
                exit_code = code if code != 0 else exit_code
                _stop_children(signal.SIGTERM)
                for _, q in procs:
                    if q.poll() is None:
                        try:
                            q.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            q.kill()
                raise SystemExit(exit_code if exit_code != 0 else code)
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("[situation_stack] interrupt, stopping children...", file=sys.stderr)
        _stop_children(signal.SIGINT)
    finally:
        for name, p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        for name, p in procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    print(f"[situation_stack] kill {name} (pid {p.pid})", file=sys.stderr)
                    p.kill()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass


if __name__ == "__main__":
    main()
