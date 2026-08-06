"""LOG-ONLY supervised demo run: the FULL chassis_sequence demo, unchanged,
with the signal tap recording live 50 Hz phase-labeled features — the source
for the live envelope (envelope_build --from-signals).

ik_demo stays untouched: run_item calls mover.place()/pick()/pick_gated(), so
a SuctionMover SUBCLASS injects the phase labels + the descent tick_cb. The
Supervisor is forced to LOG-ONLY (envelope_path=None): nothing can abort,
whatever envelope.json says — this launcher exists to COLLECT nominal data.
(Armed operation is the run_item integration in the README, not this script.)

    python -m LGES.ik_VLM.run_supervised [--gripper] [--auto-move] [--dashboard]

Same flags/prompts as `python -m LGES.ik_demo.chassis_sequence`. Output:
LGES/ik_VLM/logs/signals_<timestamp>.jsonl — one file per run. Only feed
CLEAN (nominal) runs to envelope_build; delete/skip a run's file if the run
had failures or interventions.
"""

from __future__ import annotations

import sys

from loguru import logger

from ..ik_demo import chassis_sequence as cs
from ..ik_demo import config as ikcfg
from ..ik_demo.drivers import suction_io
from ..ik_demo.gripper import GripperMover
from ..ik_demo.suction import SuctionMover
from .supervisor import Supervisor


class SupervisedSuctionMover(SuctionMover):
    """SuctionMover whose primitives label tap phases and feed the descent
    tick_cb. With a LOG-ONLY supervisor the tick_cb always returns False, so
    behavior is identical to the plain mover."""

    _sup: "Supervisor | None" = None

    def attach(self, sup: Supervisor) -> None:
        self._sup = sup

    def _phased(self, phase: str, fn):
        if self._sup is None:
            return fn(None)
        self._sup.tap.set_phase(phase)
        try:
            return fn(self._sup._tick_cb)
        finally:
            self._sup.tap.set_phase("transport")

    def place(self, pose, expected_z=None, misseat_tol_m=None, tick_cb=None):
        return self._phased("place", lambda cb: super(SupervisedSuctionMover, self).place(
            pose, expected_z=expected_z, misseat_tol_m=misseat_tol_m,
            tick_cb=tick_cb if tick_cb is not None else cb))

    def pick(self, pose, expected_z=None):
        return self._phased("pick", lambda _cb: super(SupervisedSuctionMover, self).pick(
            pose, expected_z=expected_z))

    def pick_gated(self, pose, case_center=None, expected_z=None):
        return self._phased("pick", lambda _cb: super(SupervisedSuctionMover, self).pick_gated(
            pose, case_center=case_center, expected_z=expected_z))


def _main() -> None:
    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot

    use_gripper = "--gripper" in sys.argv
    auto_move = "--auto-move" in sys.argv
    use_dashboard = "--dashboard" in sys.argv

    logger.warning("=" * 60)
    logger.warning("SUPERVISED (LOG-ONLY) chassis demo — same motion as")
    logger.warning("chassis_sequence; adds the 50 Hz signal log, aborts NOTHING.")
    logger.warning("Clear the strafe path. E-stop in reach.")
    logger.warning("=" * 60)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    suction_io.suction_off()
    configs = get_robot_config()
    configs.enable_sensor("head_camera")
    configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=configs) as bot:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            logger.warning("head camera may not be active")
        cs.set_head_pitch(bot, angle=30.0)
        publisher = None
        if use_dashboard:
            from ..ik_demo.dashboard_publish import DashboardPublisher
            publisher = DashboardPublisher(bot).start()
        try:
            with SupervisedSuctionMover(bot) as m:
                release = m.software_estop_active()
                if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                    return
                if not m.ensure_ready(release_estop=release):
                    logger.error("arm not ready — aborting")
                    return
                gripper = None
                if use_gripper:
                    gripper = GripperMover(bot)
                    gripper.ensure_ready(release_estop=release)
                    if not gripper.initialize():
                        logger.warning("gripper unavailable — divert disabled")
                        gripper = None
                sup = Supervisor(bot, m, envelope_path=None, use_vlm=False)
                m.attach(sup)
                sup.start()
                sup.tap.set_phase("transport")
                try:
                    logger.info("-> both arms safe home")
                    from ..ik_demo.go_home import both_arms_home
                    both_arms_home(bot, left=m)
                    ok = cs.run(bot, m, gripper=gripper, scan=use_gripper,
                                auto=auto_move)
                    logger.info("-> home")
                    m.move_joints(m._home_seed)
                    logger.info("sequence {}", "OK" if ok else "FAILED")
                finally:
                    sup.stop()
        finally:
            if publisher is not None:
                publisher.stop()


if __name__ == "__main__":
    _main()
