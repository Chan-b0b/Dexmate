"""Offline view-park feasibility scan (headless ArmMover, torso = cfg.TORSO_JOINTS).

Down-pointing EE (GRASP_ORIENTATION_RPY), home-seeded solves — mirrors the
_view_park home-seed retry path. Prints a feasibility map per z and checks the
candidate points tried today.
"""
import sys
sys.path.insert(0, "/home/dexmate/LGES/Dexmate")

import numpy as np
from LGES.ik_demo.arm import ArmMover
from LGES.ik_demo import config as cfg

m = ArmMover(robot=None, side="left")
rpy = tuple(cfg.GRASP_ORIENTATION_RPY)
print(f"torso_q (cfg.TORSO_JOINTS) = {np.round(m._torso_q, 4).tolist()}")
print(f"reach tol = {cfg.REACH_TOL_M*1000:.0f}mm, rpy = straight-down\n")

def ok(x, y, z):
    sol = m.solve_pose((x, y, z), rpy, seed=m._home_seed, min_motion=False)
    good = sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits and not sol.in_collision
    return good, sol.pos_err_m

for z in (1.10, 1.15, 1.20, 1.25):
    print(f"z = {z:.2f}   (rows: y left->  cols: x fwd 0.60..1.00)")
    ys = np.arange(0.25, 0.551, 0.05)
    xs = np.arange(0.60, 1.001, 0.05)
    print("      " + " ".join(f"{x:４.2f}" if False else f"{x:5.2f}" for x in xs))
    for y in ys[::-1]:
        row = []
        for x in xs:
            good, err = ok(float(x), float(y), float(z))
            row.append("  OK " if good else f"{min(err*1000,999):4.0f}s")
        print(f"y={y:4.2f} " + " ".join(row))
    print()

print("today's candidates:")
for p in [(0.90, 0.40, 1.20), (0.90, 0.40, 1.25), (0.70, 0.40, 1.20),
          (0.90, 0.50, 1.15), (0.85, 0.45, 1.15)]:
    good, err = ok(*p)
    print(f"  {p}: {'REACHABLE' if good else f'short {err*1000:.1f}mm'}")
