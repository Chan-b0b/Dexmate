"""Live gCU probe for grip-onset tuning (no arm motion — gripper only).

Put the object between the fingers first (finish a jog there, or hold it by
hand — mind the pinch). The script opens the gripper, waits for ENTER, then
closes at the collection speed/force while printing EVERY status poll
(t, gPO, gCU, gOBJ). The soft-grip trigger from config
(SOFT_GRIP_CU_STOP x SOFT_GRIP_CU_CONSECUTIVE) is evaluated live and marked,
and by default it freezes exactly like collection does.

After the run a what-if table shows where every CU_STOP x CONSECUTIVE combo
would have triggered on this exact profile — pick config values from one run.

    python -m LGES.vla_training.right_gripper.probe_grip_gcu
    python -m LGES.vla_training.right_gripper.probe_grip_gcu --no-stop
        # keep closing through contact for the FULL ramp. WARNING: ends at
        # the force controller's ~20 N stall — not for crushable objects.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from LGES.ik_demo import config as ikcfg  # noqa: E402
from LGES.ik_demo.drivers.robotiq_usb import RobotiqGripperUSB  # noqa: E402
from LGES.vla_training.right_gripper import config as rcfg  # noqa: E402

_WARMUP_S = 0.15  # motor inrush — same dead time as _grip_soft
_SETTLE_S = 0.8   # keep sampling this long after motion stops / freeze


def _whatif_table(samples: list[tuple[float, int, int, int]]) -> None:
    """Where each CU_STOP x CONSECUTIVE combo would trigger on this profile."""
    stops = [3, 4, 5, 6, 8, 10]
    consecs = [1, 2, 3, 5]
    print("\nwhat-if trigger points (t[s] @ gPO), '-' = never:")
    print("  CU_STOP  " + "".join(f"x{c:<12d}" for c in consecs))
    for stop in stops:
        row = f"  {stop:7d}  "
        for consec in consecs:
            hits, fired = 0, None
            for t, po, cu, _obj in samples:
                if t <= _WARMUP_S:
                    continue
                hits = hits + 1 if cu >= stop else 0
                if hits >= consec:
                    fired = (t, po)
                    break
            row += (f"{fired[0]:5.2f}@{fired[1]:<5d} " if fired else "    -        ")
        print(row)


def main() -> None:
    no_stop = "--no-stop" in sys.argv
    g = RobotiqGripperUSB()
    if not g.initialize():
        print("gripper unavailable")
        return
    g.open()
    stop_desc = ("NONE — full ramp to the ~20 N stall"
                 if no_stop else
                 f"gCU>={rcfg.SOFT_GRIP_CU_STOP} x{rcfg.SOFT_GRIP_CU_CONSECUTIVE} "
                 f"(+{rcfg.SOFT_GRIP_SQUEEZE} squeeze), same as collection")
    input(f"Object between the fingers? ENTER closes at speed={rcfg.GRIP_SPEED:#x} "
          f"force={rcfg.GRIP_FORCE}, stop: {stop_desc} > ")

    speed, force = int(rcfg.GRIP_SPEED), int(rcfg.GRIP_FORCE)
    g._write_control(0x09, ikcfg.ROBOTIQ_CLOSE_POS, speed, force)
    t0 = time.time()
    samples: list[tuple[float, int, int, int]] = []
    hits, frozen, done_at = 0, None, None
    while time.time() - t0 < rcfg.SOFT_GRIP_TIMEOUT_S + _SETTLE_S:
        s = g.read_status()
        if s is None:
            time.sleep(0.02)
            continue
        t = time.time() - t0
        samples.append((t, s.gPO, s.gCU, s.gOBJ))
        mark = ""
        hits = hits + 1 if (t > _WARMUP_S and s.gCU >= rcfg.SOFT_GRIP_CU_STOP) else 0
        if frozen is None and hits >= int(rcfg.SOFT_GRIP_CU_CONSECUTIVE):
            frozen = (t, s.gPO)
            mark = "  <-- soft-grip trigger"
            if not no_stop:
                target = min(255, s.gPO + int(rcfg.SOFT_GRIP_SQUEEZE))
                g._write_control(0x09, target, speed, force)
                mark += f" (frozen at {target})"
        if s.gOBJ in (2, 3):
            mark += f"  [gOBJ={s.gOBJ}: {'object' if s.gOBJ == 2 else 'at target'}]"
            if done_at is None:
                done_at = t
        print(f"t={t:6.3f}s  gPO={s.gPO:3d}  gCU={s.gCU:3d} (~{s.gCU * 10:4d} mA)  "
              f"gOBJ={s.gOBJ}{mark}")
        if done_at is not None and t - done_at > _SETTLE_S:
            break
        time.sleep(0.02)

    moving = [cu for t, _po, cu, obj in samples if obj == 0 and t > _WARMUP_S
              and (frozen is None or t < frozen[0])]
    holding = [cu for t, _po, cu, _obj in samples if done_at is not None and t > done_at + 0.3]
    print(f"\nfree-travel gCU: max {max(moving, default=0)} "
          f"(~{max(moving, default=0) * 10} mA) over {len(moving)} polls")
    if frozen is not None:
        print(f"soft-grip trigger: t={frozen[0]:.3f}s at gPO={frozen[1]}")
    print(f"hold gCU after stop: {holding[-1] if holding else 'n/a'}  "
          f"final gPO: {samples[-1][1] if samples else 'n/a'}")
    _whatif_table(samples)

    input("\nENTER to open the gripper > ")
    g.open()
    g.disconnect()


if __name__ == "__main__":
    main()
