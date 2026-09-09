"""Configuration for the ik_demo case + battery suction demo.

Single-arm suction pick-and-place: move an empty case left -> right, seat two
batteries into it; barcode-matched batteries are diverted into a separate
divert case (chassis_sequence). Forward only (no undo). See PLAN.md for the
full design.

All poses are in the robot **base_link** frame, metres / radians. Orientation is
the straight-down suction approach (cup facing -Z).

This file is grown incrementally alongside the build order: it currently holds
what the drivers + arm.py (motion core) need. Feature-specific keys are added
when their module is built — descent/force in suction.py, barcode-search
geometry in barcode.py.

Split by domain 2026-09-04, when config.py passed 1200 lines: the tuning now
lives in one module per domain next to this file, and this __init__ is the
FACADE over them. Every consumer keeps doing ``from . import config as cfg``
and reading ``cfg.NAME``, so nothing outside this package changed — including
the few places that OVERRIDE a value at runtime (test_handoff), which still
land on this module.

    config/robot.py       URDF, IK budget, speed caps
    config/geometry.py    transport heights, taught poses, resolve_poses, stances
    config/sequence.py    layer loop, stack heights, phase retry
    config/suction.py     vacuum IO, descent + contact, corner seat
    config/gripper.py     right arm: Robotiq, barcode reader, divert, box pick
    config/chassis.py     chassis legs, bin align, place trim + recovery, park
    config/vla.py         FiLM authority probe
    config/lid.py         --lid
    config/show.py        standing-object show: stand_place.py

The star imports below are in dependency order; each module imports by name the
handful of cross-domain values it derives from. Add a domain by dropping a
module in here and adding one star import. Note the submodules end up as
attributes too (``cfg.chassis`` is the module) — harmless, since every config
NAME is upper case.
"""

from __future__ import annotations

from .robot import *       # noqa: F401,F403 — URDF / IK / speed caps
from .geometry import *    # noqa: F401,F403 — transport + taught poses
from .sequence import *    # noqa: F401,F403 — layer loop + stack heights
from .suction import *     # noqa: F401,F403 — cup, descent, corner seat
from .gripper import *     # noqa: F401,F403 — right arm / barcode / box
from .chassis import *     # noqa: F401,F403 — chassis legs + place tuning
from .vla import *         # noqa: F401,F403 — FiLM authority probe
from .lid import *         # noqa: F401,F403 — --lid
from .show import *        # noqa: F401,F403 — standing-object show (stand_place)
