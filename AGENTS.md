# AGENTS.md

Quick orientation for AI coding agents working in this workspace. For a deeper
project-specific reference, also read
[.github/agents/Dexmate_developer.agent.md](.github/agents/Dexmate_developer.agent.md)
and [.claude/CLAUDE.md](.claude/CLAUDE.md) (general behavioral guidelines).

## What this repo is

Robot control / manipulation code for the **Dexmate Vega 1p** humanoid (dual
7-DOF arms + 3-DOF torso + head + grippers/suction). Several loosely-coupled
sub-projects share the `dexcontrol` SDK and a few utility modules.

## Layout

| Folder | Purpose |
|---|---|
| [dexcontrol/](dexcontrol/) | The Dexmate SDK (vendored — see its [README](dexcontrol/README.md) and [`examples/`](dexcontrol/examples/)) |
| [grasp_box/](grasp_box/) | Two-finger gripper IK (pink/pinocchio), wrench-based grasping, teach/replay utilities |
| [LGES/](LGES/) | Suction-cup pick-and-place; main demo is [LGES/case_battery_demo/](LGES/case_battery_demo/) |
| [perception/](perception/) | Camera + detection helpers |
| [situation_bundle/](situation_bundle/) | Mission-level orchestration / ROS2 bridge runtime |

`grasp_box/read_force.py` and `grasp_box/utils.py` are imported across
sub-projects via `sys.path.insert` — keep their public APIs stable.

## Environment

- **Python**: 3.12 (`/opt/venv/`); deps preinstalled (dexcontrol, dexcomm,
  dexmotion, pink, pinocchio, qpsolvers, numpy, tyro, loguru, scipy).
- **URDF**: `/opt/venv/lib/python3.12/site-packages/dexmate_urdf/robots/humanoid/vega_1p/vega_1p_gripper.urdf`
- **Required env vars** (set in `~/.bashrc`): `ROBOT_NAME`, `ZENOH_CONFIG`
  (points to the workspace's `.dzcfg` file).
- The dev container is the only supported runtime; do not assume host paths.

## Running things

Most scripts are runnable directly (`python <file>.py`). The case-battery demo
is a package:

```bash
cd LGES
python -m case_battery_demo.run_demo                  # forward only
python -m case_battery_demo.run_demo --undo           # forward then reverse
python -m case_battery_demo.run_demo --loop           # repeat until Ctrl-C
```

There is no test suite; verification is manual on the live robot. Do not
auto-run any script that moves the robot — always show the user the command
and let them launch it.

## Conventions specific to this codebase

- **`config.py` per sub-project**: every magic number lives in a top-level
  `config.py` (e.g. [LGES/case_battery_demo/config.py](LGES/case_battery_demo/config.py),
  [grasp_box/config.py](grasp_box/config.py)). Don't hard-code thresholds in
  control loops; add a named constant.
- **Hardware safety prompt**: any script that moves the robot must `input()`
  a "y/N" confirmation after a `logger.warning(...)` block. Pattern is in
  [LGES/case_battery_demo/run_demo.py](LGES/case_battery_demo/run_demo.py).
- **Position-mode required**: `arm.set_joint_pos(...)` only works after
  `arm.set_modes(["position"] * 7)` AND with software E-Stop released. See
  `SuctionMover.ensure_ready()` in [LGES/case_battery_demo/grasp.py](LGES/case_battery_demo/grasp.py).
- **IK**: pink + pinocchio with `FrameTask` (EE) + `PostureTask` (centering on
  joint mid-ranges) + `ConfigurationLimit`/`VelocityLimit`. For arm-only
  motion, lock everything else via `pin.buildReducedModel` rather than relying
  on the solver to leave joints alone.
- **Wrench/force**: use `read_force.tare_force(side, robot)` then
  `get_force(side, robot)`. Note the per-axis-abs quirk in `get_force` — large
  pose changes after a tare can produce non-zero readings without contact;
  re-tare when the load on the cup/gripper changes.
- **Suction (LGES)**: HTTP `weblogic` programs at `192.168.5.1` + a socketio
  telemetry stream. The seal sensor is `dInput[0]` (DI0) — see
  `VacuumMonitor` in [LGES/case_battery_demo/suction_io.py](LGES/case_battery_demo/suction_io.py).
  Do **not** use `toolA` (pump current) as the seal signal; OFF-idle
  baseline overlaps the sealed value.
- **Logging**: `loguru` everywhere; format strings use `{}` not f-strings so
  values are lazily formatted.
- **CLI args**: `tyro.cli(Args)` over a `@dataclass`. Don't introduce
  `argparse`.
- **Smoothstep interpolation** for arm waypoint moves: `alpha = t*t*(3-2*t)`.

## Don'ts

- Don't move the torso to "help" an unreachable arm pose — reposition the
  workpiece instead (see comment at top of `LGES/case_battery_demo/config.py`).
- Don't bypass the safety confirmation, even for "quick tests".
- Don't add a test framework, package manager, linter, or CI config without
  being asked — none currently exist in this repo by design.
- Don't refactor `dexcontrol/` (vendored upstream).
