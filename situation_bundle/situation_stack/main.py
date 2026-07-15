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

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# situation_bundle/situation_stack/main.py → bundle root
_BUNDLE_ROOT = Path(__file__).resolve().parent.parent


def _popen_session_kwargs() -> dict[str, bool]:
    """POSIX: new session so the launcher can stop the whole subtree via os.killpg."""
    if sys.platform == "win32":
        return {}
    return {"start_new_session": True}


def _signal_process_group(p: subprocess.Popen, sig: int) -> None:
    if p.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            p.send_signal(sig)
        else:
            os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            p.send_signal(sig)
        except ProcessLookupError:
            pass


def _kill_process_group(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            p.kill()
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            p.kill()
        except ProcessLookupError:
            pass


def _publish_manipulation_state_failed_on_interrupt() -> None:
    """Publish ``/manipulation/state`` once with JSON ``status: failed`` (Ctrl+C path).

    Matches ``manipulation_manager`` payload shape. Best-effort; ignores failures.
    Set ``SITUATION_SKIP_INTERRUPT_MANIPULATION_STATE=1`` to disable (e.g. tests).
    """
    if os.environ.get("SITUATION_SKIP_INTERRUPT_MANIPULATION_STATE", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
    except ImportError:
        print(
            "[situation_stack] skip interrupt /manipulation/state: rclpy or std_msgs missing",
            file=sys.stderr,
        )
        return
    node = None
    try:
        rclpy.init(args=None)
        node = rclpy.create_node("situation_stack_interrupt_pub")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        pub = node.create_publisher(String, "/manipulation/state", qos)
        msg = String()
        msg.data = json.dumps(
            {
                "sequence_id": None,
                "status": "failed",
                "message": "situation_stack interrupted (Ctrl+C)",
            },
            ensure_ascii=False,
        )
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
    except Exception as e:
        print(
            f"[situation_stack] could not publish interrupt /manipulation/state: {e}",
            file=sys.stderr,
        )
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def _launch_debug_from_argv(argv: list[str]) -> tuple[list[str], bool]:
    """Strip ``--debug`` or ``--arg debug``; return remaining args and whether debug was requested."""
    debug = False
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--debug":
            debug = True
            i += 1
        elif (
            argv[i] == "--arg"
            and i + 1 < len(argv)
            and argv[i + 1].lower() == "debug"
        ):
            debug = True
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out, debug


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
            **_popen_session_kwargs(),
        )
        out.append((name, p))
        time.sleep(0.25)
    return out


def main() -> None:
    rest, debug = _launch_debug_from_argv(sys.argv[1:])
    if rest:
        print(
            "[situation_stack] note: ignoring unused launch args: "
            + " ".join(rest),
            file=sys.stderr,
        )

    env = os.environ.copy()
    _prepend_pythonpath(env, str(_BUNDLE_ROOT))
    if debug:
        env["SITUATION_LOG_LEVEL"] = "DEBUG"
        print(
            "[situation_stack] SITUATION_LOG_LEVEL=DEBUG "
            "(manipulation_manager /execution/state trace)",
            file=sys.stderr,
        )

    if _bridge_enabled(env):
        _check_imports_dexcomm()

    procs = _build_procs(env)
    if not procs:
        raise SystemExit(1)

    print(
        "[situation_stack] started: "
        + ", ".join(n for n, _ in procs)
        + " (Ctrl+C or Ctrl+Z to stop)",
        file=sys.stderr,
    )

    def _sigtstp_raises_keyboard_interrupt(_signum: int, _frame: object) -> None:
        """Ctrl+Z (SIGTSTP) → same path as Ctrl+C: failed publish + stop children."""
        raise KeyboardInterrupt

    if hasattr(signal, "SIGTSTP"):
        signal.signal(signal.SIGTSTP, _sigtstp_raises_keyboard_interrupt)

    def _stop_children(sig: int) -> None:
        for _name, p in procs:
            _signal_process_group(p, sig)

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
                            _kill_process_group(q)
                raise SystemExit(exit_code if exit_code != 0 else code)
            time.sleep(0.4)
    except KeyboardInterrupt:
        if any(name == "manipulation_manager" for name, _ in procs):
            _publish_manipulation_state_failed_on_interrupt()
        print("[situation_stack] interrupt, stopping children...", file=sys.stderr)
        _stop_children(signal.SIGINT)
    finally:
        for name, p in procs:
            _signal_process_group(p, signal.SIGINT)
        for name, p in procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    print(f"[situation_stack] kill {name} (pid {p.pid})", file=sys.stderr)
                    _kill_process_group(p)
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass


if __name__ == "__main__":
    main()
