"""Supervised execution: wraps SuctionMover primitives with the monitor +
Tier-0/1/2 recovery. The nominal path is byte-for-byte the scripted IK
sequence — the supervisor only labels phases, watches the envelope, and takes
over when it trips.

Modes
-----
- No envelope.json yet -> LOG-ONLY: the tap records signal logs (the envelope
  source) and never aborts anything. First deployments run here.
- Envelope present -> armed: an in-descent trip halts the place within one
  tick (suction.place tick_cb) with the part held; between-primitive trips
  are reported at the next phase boundary.

Integration (chassis_sequence.run_item): replace the place call —

    pres = mover.place(place_pose, expected_z=exp_z, misseat_tol_m=mtol)

with

    pres = sup.place(place_pose, label=label, station="target",
                     layers=tgt_layers, expected_z=exp_z, misseat_tol_m=mtol,
                     plane_z=tgt_plane, pose_key=pose_key)

Everything downstream (PickResult, ZTracker logging) is unchanged.

Standalone (no arm motion): `python -m LGES.ik_VLM.supervisor --tap-test`
runs the tap + monitor live while you poke the cup by hand — a safe
end-to-end check of sampling, phases, and trip thresholds.
"""

from __future__ import annotations

from loguru import logger

from . import config as cfg
from .monitor import EnvelopeModel, Monitor
from .recovery import RecoveryContext, RecoveryRunner
from .signals import SignalTap
from .vlm_advisor import VLMAdvisor


class Supervisor:
    def __init__(self, bot, mover, envelope_path: "str | None" = cfg.ENVELOPE_PATH,
                 use_vlm: bool = True, log_signals: bool = True) -> None:
        self._bot = bot
        self._mover = mover
        # envelope_path=None forces LOG-ONLY even when an envelope file exists
        model = EnvelopeModel.load(envelope_path) if envelope_path else None
        self.monitor = Monitor(model) if model is not None else None
        if self.monitor is None:
            logger.warning("[ik_VLM] no envelope at {} — LOG-ONLY mode "
                           "(no aborts; signals are recorded for envelope_build)",
                           envelope_path)
        self.tap = SignalTap(
            mover,
            on_tick=(self.monitor.observe if self.monitor is not None else None),
            log_dir=cfg.SIGNAL_LOG_DIR if log_signals else None)
        self.recovery = RecoveryRunner(bot, mover,
                                       VLMAdvisor() if use_vlm else None)

    def start(self) -> None:
        self.tap.start()

    def stop(self) -> None:
        self.tap.stop()

    def set_phase(self, phase: str) -> None:
        """Label coarse phases from the orchestrator (e.g. "transport" around
        chassis legs) so the envelope keys line up."""
        self.tap.set_phase(phase)

    # ------------------------------------------------------------------
    def _tick_cb(self, ee_z: float, _f) -> bool:
        self.tap.note_descent(ee_z)
        return self.monitor is not None and self.monitor.tripped()

    def place(self, pose, *, label: str, station: str = "target", layers: int,
              expected_z=None, misseat_tol_m=None, plane_z=None,
              pose_key: str | None = None, replan=None):
        """Supervised SuctionMover.place. Returns the (final) PickResult — on a
        successful re-entry it is the retry's result, so ZTracker records the
        real contact.

        ``replan(center) -> pose``: how to rebuild the place pose from a fresh
        detection. Default: resolve_poses(center)[pose_key] with NO trims —
        pass the caller's own replan to keep its trim policy."""
        self.tap.set_phase("place")
        res = self._mover.place(pose, expected_z=expected_z,
                                misseat_tol_m=misseat_tol_m, tick_cb=self._tick_cb)
        self.tap.set_phase("idle")
        if res.reason != "monitor_abort":
            self._boundary_check(label, station)
            return res

        # --- tripped mid-descent: part HELD, arm halted -------------------
        self.recovery.hold_and_lift()                       # Tier 0, automatic
        holder = {"res": res}

        def _retry_place(center) -> bool:
            p = self._replan(center, pose, pose_key, replan)
            if p is None:
                return False
            self.tap.set_phase("place")
            r = self._mover.place(p, expected_z=expected_z,
                                  misseat_tol_m=misseat_tol_m, tick_cb=self._tick_cb)
            self.tap.set_phase("idle")
            holder["res"] = r
            return bool(r.success)

        trip = self.monitor.trip_info()
        ctx = RecoveryContext(
            label=label, station=station, layers=layers, phase="descend",
            holding_expected=True,
            trip_reason=trip.describe() if trip else "monitor trip",
            plane_z=plane_z, expected_xy=(float(pose[0]), float(pose[1])),
            retry_place=_retry_place)
        outcome = self.recovery.run(ctx)
        self.monitor.reset()
        out = holder["res"]
        if outcome == "stopped" and not out.success:
            out.reason = "monitor_abort"
        return out

    def pick(self, pose, *, label: str, station: str = "source", layers: int,
             expected_z=None, plane_z=None, pose_key: str | None = None,
             gated: bool = False, case_center=None, replan=None):
        """Supervised pick. No in-descent hook (pick descents keep the script's
        own force/seal guards); the monitor is checked at the boundary."""
        self.tap.set_phase("pick")
        if gated:
            res = self._mover.pick_gated(pose, case_center=case_center,
                                         expected_z=expected_z)
        else:
            res = self._mover.pick(pose, expected_z=expected_z)
        self.tap.set_phase("idle")
        self._boundary_check(label, station, layers=layers, plane_z=plane_z,
                             holding_expected=res.success, pose=pose,
                             pose_key=pose_key, replan=replan)
        return res

    # ------------------------------------------------------------------
    def _replan(self, center, pose, pose_key, replan):
        if replan is not None:
            return replan(center)
        if center is None or pose_key is None:
            logger.warning("[ik_VLM] cannot replan (no detection / no pose_key) "
                           "— retrying the ORIGINAL pose")
            return pose
        from ..ik_demo.config import resolve_poses
        return resolve_poses(center)[pose_key]

    def _boundary_check(self, label: str, station: str, **ctx_kw) -> None:
        """Between-primitive trip: the primitive already ended at rest, so no
        hold is needed — report and let the operator continue or recover."""
        if self.monitor is None or not self.monitor.tripped():
            return
        trip = self.monitor.trip_info()
        logger.warning("[ik_VLM] monitor tripped OUTSIDE a descent: {}",
                       trip.describe())
        cmd = input("[ik_VLM] Enter=continue / r=recover / q=raise: ").strip().lower()
        if cmd == "r":
            ctx = RecoveryContext(
                label=label, station=station,
                layers=int(ctx_kw.get("layers", 1)),
                phase=trip.phase, holding_expected=bool(ctx_kw.get("holding_expected", False)),
                trip_reason=trip.describe(), plane_z=ctx_kw.get("plane_z"))
            self.recovery.run(ctx)
        elif cmd == "q":
            raise RuntimeError(f"ik_VLM monitor trip: {trip.describe()}")
        self.monitor.reset()


# ---------------------------------------------------------------------------
# Safe standalone check: NO arm motion. Starts the tap + monitor on the live
# robot; poke/push the cup by hand and watch trips. q + Enter to quit.
# ---------------------------------------------------------------------------
def _tap_test() -> None:
    import sys
    import time

    from dexcontrol.robot import Robot

    from ..ik_demo.suction import SuctionMover

    seconds = 60.0
    if "--seconds" in sys.argv:
        seconds = float(sys.argv[sys.argv.index("--seconds") + 1])
    logger.info("tap test: NO arm motion — sampling {:.0f}s; push on the cup "
                "to exercise the monitor", seconds)
    with Robot() as bot:
        m = SuctionMover(bot)
        sup = Supervisor(bot, m, use_vlm=False)
        m.tare()
        sup.tap.set_phase("descend")   # arm the strictest phase for the hand test
        sup.start()
        try:
            t_end = time.time() + seconds
            while time.time() < t_end:
                time.sleep(0.5)
                ticks = sup.tap.recent(0.5)
                if ticks:
                    t = ticks[-1]
                    print(f"\r f_ax={t.f_ax:6.2f}N f_lat={t.f_lat:6.2f}N "
                          f"t_mag={t.t_mag:5.2f}Nm df={t.df_mag:7.1f}N/s "
                          f"q_err={t.q_err_max:5.3f}rad "
                          f"tripped={sup.monitor.tripped() if sup.monitor else '-'}   ",
                          end="")
        finally:
            print()
            sup.stop()


if __name__ == "__main__":
    _tap_test()
