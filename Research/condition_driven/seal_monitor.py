#!/usr/bin/env python3
"""Live vacuum-seal signal monitor — verify the seal sensor.

Connects to the SAME DI0 socketio source the recorder / run_policy use
(case_battery_demo.suction_io.VacuumMonitor) and continuously prints:

  DI0          raw physical seal sensor (dInput[0]) -- THIS is the sensor to verify
  is_sealed()  what the policy/recorder log as `vacuum_sealed`
               = DI0 AND we-commanded-suction-ON in THIS process
  suction_cmd  whether suction has been commanded on in this process
  toolA        pump current (diagnostic only; intentionally NOT used for seal)

GOTCHA: is_sealed() is GATED on suction_io.is_suction_commanded_on(). If you make
a seal by hand WITHOUT commanding suction, is_sealed() stays False while DI0 goes
True. So to check the SENSOR, watch DI0; is_sealed() is what gets logged during a
real rollout (where suction is commanded on, so is_sealed() == DI0).

Read-only: commands no suction/blow, moves nothing.

Run with the demo venv (needs socketio/requests):
  /home/dexmate/vla_venv/bin/python seal_monitor.py
"""
import sys
import time
from pathlib import Path

# make `case_battery_demo` importable (same as run_policy.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LGES"))
from case_battery_demo import suction_io  # noqa: E402


def main():
    mon = suction_io.VacuumMonitor()
    mon.start()
    print("connecting to DI0 source...  (Ctrl-C to stop)\n")
    try:
        while True:
            di0 = bool(getattr(mon, "_di0", False))
            print(f"\rconn={'Y' if mon.is_connected() else '.':1}  "
                  f"DI0={'SEAL' if di0 else '----'}  "
                  f"is_sealed={'T' if mon.is_sealed() else 'F'}  "
                  f"suction_cmd={'ON ' if suction_io.is_suction_commanded_on() else 'off'}  "
                  f"toolA={mon.get_tool_current():.4f}A   ",
                  end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        mon.stop()


if __name__ == "__main__":
    main()
