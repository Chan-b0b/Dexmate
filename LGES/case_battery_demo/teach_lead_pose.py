"""Teach-mode helper: lead the arm by hand (compliant) and record EE + joint poses.

Enables a current-based admittance ("lead-through") loop so you can physically
guide the arm. The arm stays actively controlled — it never goes limp and holds
itself against gravity. Press ENTER to record the current pose; each record
appends one EE line (x y z r p y, base_link) and one joint line (7 angles) to
taught_ee_poses / taught_joint_poses, matching teach_pose.py's format.

    python -m case_battery_demo.teach_lead_pose                # default side (cfg.ARM_SIDE)
    python -m case_battery_demo.teach_lead_pose --side right   # gripper arm

Commands (type, then ENTER):
    <empty>  : record current pose (EE + joints)
    + / =    : more compliant (easier to push)      -  : stiffer
    ] / [    : more / less sensitive (deadband)
    } / {    : more / less lead range (authority in high-load poses)
    d        : toggle the live offset readout
    q        : quit

Input is line-based (input()) rather than raw single-key so it works in the
VSCode integrated terminal AND the Run/Debug console — raw-mode key reads need a
real TTY and silently read nothing under the Run button. The 200 Hz compliance
loop runs in a background thread so the blocking input() never stalls control.

Control law: memoryless proportional admittance — each cycle commands the ACTUAL
joint position plus a force-proportional standing LEAD in the push direction
(bounded by MAX_LEAD). A standing lead (not a vel*dt step, which at 200 Hz is
below joint stiction) is what makes it move. No accumulated setpoint (no
integrator) => no limit-cycle wobble. The baseline always adapts, tracking the
holding current so its deviation reads as your push.

Note on why there's no gravity model: this arm's holding current is dominated by
large hysteretic stiction (~±1.5 A at the shoulder, as big as the gravity term
itself), so holding current is NOT a function of pose — a gravity feedforward
can't help. The always-adapting baseline instead just tracks whatever the
current friction/gravity state is, and the deadband rejects its residual ripple.
The residual cost is a mild spring-back when you release a sustained push.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import tyro
from dexcomm.utils import RateLimiter
from loguru import logger

from dexcontrol.robot import Robot

from .grasp import ArmMover
from . import config as cfg

CONTROL_HZ = 200.0
# Standing position lead (rad) commanded per amp of deadbanded push current, PER
# JOINT. We command actual+lead each cycle (NOT actual+vel*dt): at 200 Hz vel*dt
# is a sub-0.003 rad error the joint's stiction swallows, so the arm won't move.
# A force-proportional standing lead clears stiction even for a gentle push.
# Per-joint because joint currents differ ~30x (shoulder ~4A vs wrist ~0.1A): a
# single gain under-drives the wrists. Higher = easier to push; +/- scales all.
GAIN = np.array([0.10, 0.10, 0.12, 0.12, 0.35, 0.35, 0.35])
# Max standing lead (rad) per joint — bounds the position error the controller
# chases (safety/speed clamp). Distal joints get a bit more so they can still
# move against gravity in high-load poses. If a joint stalls "above a certain
# angle", raise its entry with } (or it may be motor-saturated / at a URDF
# limit, which no lead fixes).
MAX_LEAD = np.array([0.10, 0.10, 0.10, 0.10, 0.12, 0.12, 0.12])
# Low-pass time constant (s) for the measured current, to reject sensor noise.
CURRENT_TAU = 0.15
# Per-joint deadband (A): current offset below this is treated as noise, not a
# push. Set just above the at-rest offset ripple measured with `d` on this arm
# (~0.04 on J1/J2, ~0.02 on J3/J4, <0.01 on the wrists); real pushes are 0.2-2.5.
FORCE_THRESHOLDS = np.array([0.06, 0.08, 0.06, 0.06, 0.03, 0.03, 0.03])
# EMA time constant (s) for tracking the holding-current baseline. Always
# adapting keeps the loop stable (absorbs motion/friction current so it can't
# self-drive). Faster = settles/holds sooner but stiffer on slow sustained
# pushes. 0.5 is the original's value.
BASELINE_TAU = 0.5


def _ext_current(offset: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Deadbanded external current -> desired-motion drive (continuous at edge).

    Below the deadband: zero (rest). Beyond it: the *excess* over the deadband,
    negated so the arm moves the way you're pushing (reduces the joint's
    resisting current). Using the excess (not the raw offset) makes motion ramp
    from zero at the deadband edge — no jump — so it feels soft to start.
    """
    excess = np.maximum(np.abs(offset) - thresholds, 0.0)
    return -np.sign(offset) * excess


def main(side: str = cfg.ARM_SIDE) -> None:
    ee_frame = cfg.EE_FRAME if side == cfg.ARM_SIDE else cfg.GRIPPER_EE_FRAME
    suffix = "" if side == cfg.ARM_SIDE else f"_{side}"
    pose_file = Path(__file__).parent / f"taught_ee_poses{suffix}.txt"
    joint_file = Path(__file__).parent / f"taught_joint_poses{suffix}.txt"

    logger.warning("=" * 70)
    logger.warning("SAFETY: {} arm will become COMPLIANT (lead-through).", side)
    logger.warning("It stays actively controlled — never goes limp.")
    logger.warning("Move slowly and keep the e-stop within reach!")
    logger.warning("=" * 70)
    if input("Start? [y/N]: ").strip().lower() != "y":
        return

    dt = 1.0 / CONTROL_HZ
    alpha = np.exp(-dt / BASELINE_TAU)
    alpha_cur = np.exp(-dt / CURRENT_TAU)

    # Shared with the input thread. Dict-entry read/write is atomic under the
    # GIL, which is all the coordination this teach tool needs.
    p = {"gain": GAIN.copy(), "thresholds": FORCE_THRESHOLDS.copy(),
         "max_lead": MAX_LEAD.copy(), "debug": True}
    stop = threading.Event()
    record_req = threading.Event()
    n_recorded = 0

    with Robot() as bot:
        mover = ArmMover(bot, side=side, ee_frame=ee_frame,
                         trace=getattr(cfg, "TRACE_ENABLED", False))

        if mover.software_estop_active():
            logger.warning("Software E-Stop is ACTIVE — the arm cannot move until released.")
            if input("Release software E-Stop and enable the arm? [y/N]: ").strip().lower() != "y":
                logger.info("Leaving E-Stop engaged; exiting.")
                return
            release = True
        else:
            release = False
        if not mover.ensure_ready(release_estop=release):
            logger.error("Arm not ready (E-Stop still active?). Aborting.")
            return

        arm = mover._arm
        logger.info("ARM_SIDE={}, EE_FRAME={}", side, ee_frame)

        def control_loop() -> None:
            nonlocal n_recorded
            baseline = np.array(arm.get_joint_current(), dtype=float)
            cur_f = baseline.copy()      # low-pass-filtered current
            rate = RateLimiter(rate_hz=CONTROL_HZ)
            tick = 0

            while not stop.is_set():
                gain = p["gain"]
                thresholds = p["thresholds"]
                max_lead = p["max_lead"]

                actual = np.array(arm.get_joint_pos(), dtype=float)
                cur_f = alpha_cur * cur_f + (1.0 - alpha_cur) * np.array(
                    arm.get_joint_current(), dtype=float)
                offset = cur_f - baseline

                # Memoryless proportional admittance: command a standing lead
                # ahead of the ACTUAL position, sized by push force. No integrator
                # => no limit-cycle wobble. The lead clears joint stiction so
                # gentle pushes move it; once you stop pushing the drive falls to
                # zero and cmd == actual, so it holds where you leave it.
                lead = np.clip(gain * _ext_current(offset, thresholds), -max_lead, max_lead)
                arm.set_joint_pos((actual + lead).tolist())

                # Always adapt the baseline. Freezing it while pushed self-drives:
                # the lead moves the joint, the motion draws current, that current
                # keeps the offset over the deadband, so the lead keeps driving ->
                # runaway. Always-adapting absorbs that current within ~TAU, so the
                # arm settles when you stop pushing.
                baseline = alpha * baseline + (1.0 - alpha) * cur_f

                # Live readout: |offset| vs deadband, '*' = over (registering a
                # push). Use it to set thresholds just above the at-rest ripple.
                tick += 1
                if p["debug"] and tick % int(CONTROL_HZ // 2) == 0:
                    flags = "".join("*" if abs(o) >= t else "." for o, t in zip(offset, thresholds))
                    off = " ".join(f"{o:+.3f}" for o in offset)
                    print(f"\r[off {flags}] {off}  (over='*')   ", end="", flush=True)

                if record_req.is_set():
                    record_req.clear()
                    q = np.array(arm.get_joint_pos(), dtype=float)
                    pos, rpy = mover.current_ee_pose()
                    ee_line = " ".join(f"{v:.6f}" for v in (*pos, *rpy))
                    joint_line = " ".join(f"{v:.6f}" for v in q)
                    with pose_file.open("a") as f:
                        f.write(ee_line + "\n")
                    with joint_file.open("a") as f:
                        f.write(joint_line + "\n")
                    n_recorded += 1
                    print(f"\n>>> #{n_recorded} EE   : {ee_line}")
                    print(f"           joints: {joint_line}")
                    print(f"           saved to {pose_file.name} / {joint_file.name}")

                rate.sleep()

        worker = threading.Thread(target=control_loop, daemon=True)
        worker.start()

        print(__doc__)
        print("Compliance active. Type a command and press ENTER "
              "(empty = record, q = quit).")
        try:
            while not stop.is_set():
                cmd = input("> ").strip()
                if cmd == "":
                    record_req.set()
                elif cmd in ("q", "quit"):
                    break
                elif cmd in ("+", "="):
                    p["gain"] = np.minimum(p["gain"] * 1.5, 5.0)
                    print(f"gain={np.round(p['gain'], 2)} (more compliant)")
                elif cmd == "-":
                    p["gain"] = np.maximum(p["gain"] / 1.5, 0.02)
                    print(f"gain={np.round(p['gain'], 2)} (stiffer)")
                elif cmd == "]":
                    p["thresholds"] = np.maximum(p["thresholds"] * 0.75, 0.01)
                    print(f"deadband={np.round(p['thresholds'], 3)} (more sensitive)")
                elif cmd == "[":
                    p["thresholds"] = p["thresholds"] / 0.75
                    print(f"deadband={np.round(p['thresholds'], 3)} (less sensitive)")
                elif cmd == "}":
                    p["max_lead"] = np.minimum(p["max_lead"] * 1.5, 2.0)
                    print(f"max_lead={np.round(p['max_lead'], 3)} (more range/authority)")
                elif cmd == "{":
                    p["max_lead"] = np.maximum(p["max_lead"] / 1.5, 0.02)
                    print(f"max_lead={np.round(p['max_lead'], 3)} (less range)")
                elif cmd == "d":
                    p["debug"] = not p["debug"]
                    print(f"debug readout {'ON' if p['debug'] else 'OFF'}")
                else:
                    print("commands: <empty>=record  +/-=compliance  [/]=deadband  "
                          "}/{=lead-range  d=debug  q=quit")
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            stop.set()
            worker.join(timeout=2.0)

    logger.info("Done. {} poses recorded to {} / {}",
                n_recorded, pose_file.name, joint_file.name)


if __name__ == "__main__":
    tyro.cli(main)
