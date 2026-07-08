"""Offline reach sweep — NO robot. For a fixed (y, z) and grasp orientation,
sweep x through the IK and report which x are reachable, so we can see the arm's
forward reach limit at that side/height. Pure kinematics on the URDF at the
taught torso stance (cfg.TORSO_JOINTS).

    python -m LGES.ik_demo.reach_sweep --y 0.0926 --z 0.811 --yaw 1.9605
    python -m LGES.ik_demo.reach_sweep --y 0.5112 --z 0.810   # taught side, for comparison

"reachable" = FK error within cfg.REACH_TOL_M, in joint limits, no self-collision
(the same bar move_ee uses to accept a motion).
"""

from __future__ import annotations

import argparse

import numpy as np

from . import config as cfg
from .arm import ArmMover


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("--yaw", type=float, default=cfg.GRASP_YAW,
                    help="EE yaw (rad); default = GRASP_YAW (case yaw 0)")
    ap.add_argument("--x0", type=float, default=0.50)
    ap.add_argument("--x1", type=float, default=1.15)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    roll, pitch = cfg.GRASP_ORIENTATION_RPY[0], cfg.GRASP_ORIENTATION_RPY[1]
    arm = ArmMover(robot=None)  # torso at cfg.TORSO_JOINTS
    print(f"sweep x at y={args.y:+.4f} z={args.z:.4f} rpy=({roll:.3f},{pitch:.3f},{args.yaw:.3f}) "
          f"| REACH_TOL={cfg.REACH_TOL_M*1000:.0f}mm")

    ok_xs = []
    for x in np.arange(args.x0, args.x1 + 1e-9, args.step):
        sol = arm.solve_pose((float(x), args.y, args.z), (roll, pitch, args.yaw))
        reachable = (sol.pos_err_m <= cfg.REACH_TOL_M) and sol.in_limits and not sol.in_collision
        if reachable:
            ok_xs.append(float(x))
        print(f"x={x:.3f}  err={sol.pos_err_m*1000:6.1f}mm  in_lim={int(sol.in_limits)} "
              f"col={int(sol.in_collision)}  -> {'REACH' if reachable else '-'}")

    if ok_xs:
        print(f"\nREACHABLE x in [{min(ok_xs):.3f}, {max(ok_xs):.3f}] m  "
              f"at y={args.y:+.4f}, z={args.z:.4f}")
    else:
        print(f"\nNo reachable x in [{args.x0},{args.x1}] at y={args.y:+.4f}, z={args.z:.4f}")


if __name__ == "__main__":
    main()
