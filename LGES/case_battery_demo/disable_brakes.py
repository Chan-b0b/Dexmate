"""Release (disable) or re-engage the arms' physical motor brakes on the Vega.

The physical brakes are the mechanical holding brakes inside each arm joint.
Releasing them ("brake release" / over-limit drag mode) lets you drag the arm by
hand, including beyond the normal software joint limits — useful for manual
repositioning, calibration, or freeing a wedged arm.

    python -m case_battery_demo.disable_brakes                 # release BOTH arms' brakes
    python -m case_battery_demo.disable_brakes --side left     # one arm only
    python -m case_battery_demo.disable_brakes --engage        # re-engage (hold) the brakes

!!! DANGER !!!
Releasing the brakes makes the arm LIMP. Unlike teach_lead_pose (which keeps the
arm actively controlled and holding itself against gravity), a brake-released arm
has NOTHING holding it up — it WILL drop under its own weight and can pinch or
crush. Physically support the arm before releasing, keep the e-stop within reach,
and re-engage the brakes (--engage) the moment you are done.

The physical brakes live only on the two 7-DOF arms; the torso, head and chassis
have no releasable brake service, so this only touches the arm(s).
"""

from __future__ import annotations

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def _sides(side: str) -> list[str]:
    if side == "both":
        return ["left", "right"]
    if side in ("left", "right"):
        return [side]
    raise SystemExit(f"--side must be left, right or both (got {side!r})")


def main(side: str = "both", engage: bool = False) -> None:
    """Release or re-engage the physical arm brakes.

    Args:
        side: which arm(s) — "left", "right", or "both".
        engage: re-engage (hold) the brakes instead of releasing them.
    """
    sides = _sides(side)
    release = not engage
    verb = "release" if release else "engage"

    logger.warning("=" * 70)
    logger.warning("PHYSICAL BRAKE {} — arm(s): {}", verb.upper(), ", ".join(sides))
    if release:
        logger.warning("The arm(s) will go LIMP and DROP under gravity.")
        logger.warning("Support the arm by hand NOW and keep the e-stop within reach.")
    logger.warning("=" * 70)
    if input(f"Type 'yes' to {verb} the brakes: ").strip().lower() != "yes":
        logger.info("Aborted; brakes unchanged.")
        return

    with Robot() as bot:
        estop = getattr(bot, "estop", None)
        if release and estop is not None and estop.is_software_estop_enabled():
            # Fail-safe brakes clamp when the motors lose power, and the software
            # E-Stop cuts that power — so the brakes can't be released while it is
            # engaged. Release the E-Stop first, with explicit consent.
            logger.warning("Software E-Stop is ACTIVE — motors are unpowered, so the "
                           "brakes cannot be released while it is engaged.")
            if input("Deactivate the software E-Stop first? [y/N]: ").strip().lower() != "y":
                logger.info("Leaving E-Stop engaged; aborting.")
                return
            estop.deactivate()

        for s in sides:
            arm = getattr(bot, f"{s}_arm")
            resp = arm.release_brake(enable=release)
            ok = bool(resp.get("success"))
            (logger.info if ok else logger.error)(
                "{} arm brake {}: {}", s, verb,
                resp.get("message", "no message"))
            status = arm.get_brake_status()
            logger.info("{} arm brake-release enabled={} joints={}",
                        s, status.get("enabled"), status.get("joints"))


if __name__ == "__main__":
    tyro.cli(main)
