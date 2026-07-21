"""Loop-based micro-task data collector for VLA training.

A separate data-collection process from run_demo. Instead of running the whole
case+battery choreography once per take, this loops ONE micro-movement of ONE
sub-task many times — recording only the forward stroke each cycle — so a single
motion is harvested fast. The object never has to be re-localized: x, y stay
frozen at a taught anchor (only small jitter between cycles) and the arm just
repeats the stroke, so the object stays put on the surface the whole time.

It reuses, unmodified, the demo's motion primitives (SuctionMover, including the
`approach_to_contact` / `seal_at_contact` split that pick() now uses), the
episode recorder (RecordController) and the 15 Hz observation sampler
(DashboardPublisher), so each take lands byte-identical to run_demo's recordings.

Run from LGES/ as a module:

    python -m case_battery_demo.collect_microtasks --task battery_1_pick --step approach --cycles 50
    python -m case_battery_demo.collect_microtasks --task case_pick --step reach-hover --cycles 30
    python -m case_battery_demo.collect_microtasks --task battery_1_pick --step scan-read --cycles 40
    python -m case_battery_demo.collect_microtasks --task battery_2_pick --step seal --cycles 30
    python -m case_battery_demo.collect_microtasks --task case_pick --step approach --dry-run   # motion only

Each forward stroke is one episode; meta.json carries BOTH the micro-step
instruction and the parent sub-task instruction (set_meta_extra), so the data
can be trained per-micro OR concatenated per-parent offline.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot

# perception/ holds the shared set_head_pitch helper; expose it for import.
_PERCEPTION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "perception")
if _PERCEPTION_DIR not in sys.path:
    sys.path.insert(0, _PERCEPTION_DIR)
from utils import set_head_pitch  # noqa: E402

from . import bcr
from . import config as cfg
from . import suction_io
from .dashboard.publisher import DEFAULT_SPOOL_DIR, DashboardPublisher
from .dashboard.recorder import RecordController
from .grasp import Pose, SuctionMover
from .home_pose import go_to_default_pose

# Each pick sub-task and the taught source pose its micro-steps anchor at.
# ``thing`` fills the per-step instruction templates; ``scan`` gates scan_read.
# ``parent`` is the parent sub-task instruction the deployed policy trained on
# (run_policy.py canonical, NOT cfg.PHASE_INSTRUCTIONS — which still carries the
# pre-2026-06-16 INVERTED battery left/right labels). Stamped into meta.json so
# the data can be concatenated per-parent offline.
# ``place_dst`` is where this thing gets placed (its taught destination). The
# reach_hover loop starts from there lifted to transport z — the realistic
# "init_pos" the arm sits at after the previous place — and reaches over to the
# pick source (anchor).
TASKS = {
    "case_pick":      dict(anchor="CASE_PICK", place_dst="CASE_PLACE_R", thing="the case",
                           scan=False, parent="pick up the case with the suction cup"),
    "battery_1_pick": dict(anchor="BAT_SRC_1", place_dst="BAT_SLOT_1",  thing="the right battery",
                           scan=True,  parent="Pick up the right battery with the suction cup"),
    "battery_2_pick": dict(anchor="BAT_SRC_2", place_dst="BAT_SLOT_2",  thing="the left battery",
                           scan=True,  parent="Pick up the left battery with the suction cup"),
}

# Draft instructions — these become the policy's task tokens, so wordsmith via
# --instruction. {thing} is filled from TASKS.
STEP_INSTR = {
    "reach_hover": "move the suction cup above {thing}",
    "approach":    "lower the suction cup onto {thing}",
    "scan_read":   "read the barcode on {thing}",
    "seal":        "grab {thing} with the suction cup",
    "lift":        "lift {thing}",
}


def _taught_pose(name: str) -> Pose:
    """Build a Pose from a taught config entry (3- or 6-tuple)."""
    vals = cfg.TAUGHT_POSES.get(name)
    if vals is None:
        raise SystemExit(f"Taught pose {name!r} is not set — capture it with teach_pose.py")
    if len(vals) == 6:
        return Pose(pos=np.array(vals[:3], dtype=float), rpy=np.array(vals[3:], dtype=float))
    return Pose.from_xyz(vals)


@dataclass
class StepCtx:
    mover: SuctionMover
    pose: Pose             # anchor (x, y, contact z) + approach rpy
    hover_z: float         # anchor z + HOVER_HEIGHT_M
    scan_floor_z: float    # anchor z + BCR_SCAN_FLOOR_OFFSET_M (barcode standoff)
    reach_start: np.ndarray  # place-dst (x, y) at transport z — reach_hover origin
    rng: np.random.Generator
    xy_jitter: float       # small per-cycle x,y jitter (approach/seal)
    reach_scatter: float   # large x,y scatter for the reach start
    recorder: RecordController | None


def _jit(ctx: StepCtx, scale: float) -> tuple[float, float]:
    return float(ctx.rng.uniform(-scale, scale)), float(ctx.rng.uniform(-scale, scale))


def _tare(ctx: StepCtx) -> None:
    """Zero the wrench at the current resting pose so get_vertical_force reads
    presses relative to it. Call it hands-free before an empty-cup approach
    (like the demo's pick()), or WITH a held object before lowering it so only
    the touchdown reaction — not the held weight — trips contact (like place()).
    Without any tare the baseline defaults to 0 and get_vertical_force returns
    the raw ~14 N gravity offset, so the descent false-triggers immediately.
    grasp.py already put grasp_box on sys.path, so read_force imports."""
    from read_force import tare_force
    tare_force(cfg.ARM_SIDE, ctx.mover._robot, rotation=ctx.mover._ee_rotation())


# ── reach_hover: drive from a scattered start to the hover point ──────────────
# Object: untouched (the cup never descends to it).

def reach_reset(ctx: StepCtx, cycle: int) -> bool:
    """Reposition to a jittered start at the place destination (not recorded) —
    where the arm sits after the previous place, the reach's real init_pos."""
    dx, dy = _jit(ctx, ctx.reach_scatter)
    start = np.array([ctx.reach_start[0] + dx, ctx.reach_start[1] + dy, ctx.reach_start[2]])
    ctx.mover._move_ee_to(start, ctx.pose.rpy, cfg.MOVE_DURATION_S)
    ctx.mover._wait_until_arrived(start, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
    return True


def reach_forward(ctx: StepCtx, cycle: int) -> bool:
    """Reach from the scattered start to hover above the anchor (recorded).

    Demo two-leg: travel sideways to above the anchor at transport z, then
    descend to hover_z (same pattern pick() uses)."""
    transit = np.array([ctx.pose.pos[0], ctx.pose.pos[1], cfg.SAFE_TRANSPORT_Z])
    ctx.mover._move_ee_to(transit, ctx.pose.rpy, cfg.MOVE_DURATION_S)
    ctx.mover._wait_until_arrived(transit, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
    ctx.mover._cartesian_z_to(ctx.hover_z, ctx.pose.rpy)
    return True


# ── approach: descend (suction off) until the cup touches the part ────────────
# Object: untouched apart from a light <5N kiss; stays on the surface.

def approach_reset(ctx: StepCtx, cycle: int) -> bool:
    """Lift back to hover at a small x,y jitter (not recorded)."""
    suction_io.suction_off()
    dx, dy = _jit(ctx, ctx.xy_jitter)
    start = np.array([ctx.pose.pos[0] + dx, ctx.pose.pos[1] + dy, ctx.hover_z])
    ctx.mover._move_ee_to(start, ctx.pose.rpy, cfg.APPROACH_DESCENT_S)
    ctx.mover._wait_until_arrived(start, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
    return True


def approach_forward(ctx: StepCtx, cycle: int) -> bool:
    """Descend from the jittered hover start to force contact (recorded)."""
    _tare(ctx)  # hands-free at hover, before the force descent
    start = ctx.mover._current_ee_pos()
    res = ctx.mover.approach_to_contact(start, ctx.pose.pos[2], ctx.pose.rpy)
    if not res.success:
        logger.warning("[micro] approach cycle {}: {} (no contact)", cycle, res.trigger)
    return res.success


# ── scan_read: read the barcode from the near-contact scan floor ──────────────
# Object: untouched (1 cm standoff). Reuses the demo's scan+spiral logic.

def scan_reset(ctx: StepCtx, cycle: int) -> bool:
    """Descend to the scan floor over the anchor (not recorded). No x,y jitter:
    the spiral search inside scan_read already sweeps x,y."""
    suction_io.suction_off()
    floor = np.array([ctx.pose.pos[0], ctx.pose.pos[1], ctx.scan_floor_z])
    ctx.mover._move_ee_to(floor, ctx.pose.rpy, cfg.APPROACH_DESCENT_S)
    ctx.mover._wait_until_arrived(floor, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
    return True


def scan_forward(ctx: StepCtx, cycle: int) -> bool:
    """Scan at the floor; on no-read lift + spiral-search + return (recorded).

    Because the arm is already at the scan floor, _scan_descend_and_search's
    leading descent is a no-op (distance < 1e-4), so the recorded stroke starts
    at near-contact, not hover — as required."""
    if ctx.recorder is not None:
        ctx.recorder.set_barcode_confirmed(False)
    scanner = bcr.BackgroundScanner()
    code = ctx.mover._scan_descend_and_search(ctx.pose, scanner)
    if code is not None and ctx.recorder is not None:
        ctx.recorder.set_barcode_confirmed(True)
    if code is None:
        logger.warning("[micro] scan_read cycle {}: no barcode read", cycle)
    return code is not None


# ── seal: turn suction on at contact and wait for the vacuum seal ─────────────
# Object: stays on the surface — sealed in place, never lifted, then released.

def seal_reset(ctx: StepCtx, cycle: int) -> bool:
    """Release, clear up, travel to hover above the anchor, then re-approach to
    contact (not recorded). The lateral move to the anchor x,y is essential —
    the object was sealed in place and the arm must descend back onto IT, not
    wherever it happened to be (e.g. the home pose on cycle 0)."""
    suction_io.suction_off()
    time.sleep(0.3)
    if ctx.mover._current_ee_pos()[2] < ctx.hover_z:
        ctx.mover.lift(ctx.hover_z)  # clear the part before any lateral move
    dx, dy = _jit(ctx, ctx.xy_jitter)
    above = np.array([ctx.pose.pos[0] + dx, ctx.pose.pos[1] + dy, ctx.hover_z])
    ctx.mover._move_ee_to(above, ctx.pose.rpy, cfg.APPROACH_DESCENT_S)
    ctx.mover._wait_until_arrived(above, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
    _tare(ctx)  # hands-free above the part, before the re-approach force descent
    start = ctx.mover._current_ee_pos()
    res = ctx.mover.approach_to_contact(start, ctx.pose.pos[2], ctx.pose.rpy)
    if not res.success:
        logger.warning("[micro] seal cycle {}: re-approach failed ({})", cycle, res.trigger)
    return res.success


def seal_forward(ctx: StepCtx, cycle: int) -> bool:
    """At contact: suction on, hold, wait for the vacuum to seal (recorded)."""
    vac = suction_io.VacuumMonitor()
    vac.start()
    try:
        res = ctx.mover.seal_at_contact(ctx.pose.rpy, vac)  # owns pre-lift + suction on
    finally:
        vac.stop()
    if not res.success:
        logger.warning("[micro] seal cycle {}: {}", cycle, res.trigger)
    return res.success


# ── lift: carry the grabbed object straight up to transport height ────────────
# Object: lifted off the surface, then returned to the anchor and re-grabbed.

def lift_reset(ctx: StepCtx, cycle: int) -> bool:
    """Set the object back at the anchor and re-grab (not recorded).

    Lower with the same force-guarded descent as the approach (velocity ramps
    down near the target, halts on contact) so the object is set down softly
    instead of slammed to a fixed z. The lower needs a PAYLOAD-AWARE tare —
    taken while the object is still held — so the held weight is zeroed and only
    the touchdown reaction trips the contact threshold (exactly what the demo's
    place() does). Then release, re-tare hands-free, and seal to re-grab.
    Handles both a held object (sets it down) and cycle 0 (grabs the resting
    object: cup empty, tare is hands-free, descent lands on the object's top)."""
    above = np.array([ctx.pose.pos[0], ctx.pose.pos[1], ctx.hover_z])
    ctx.mover._move_ee_to(above, ctx.pose.rpy, cfg.APPROACH_DESCENT_S)
    ctx.mover._wait_until_arrived(above, cfg.MOVE_ARRIVAL_TOL_M, cfg.MOVE_ARRIVAL_TIMEOUT_S)
    _tare(ctx)  # WITH the held object -> baseline includes its weight, so the
                # gentle lower halts on touchdown, not on the held weight
    start = ctx.mover._current_ee_pos()
    ctx.mover.approach_to_contact(start, ctx.pose.pos[2], ctx.pose.rpy)  # soft set-down
    suction_io.suction_off()
    time.sleep(0.3)
    # Lift a bit clear of the released object, then descend back to the original
    # Z with the gentle force-guarded approach for a clean re-grab.
    cur = ctx.mover._current_ee_pos()
    ctx.mover.lift(cur[2] + 0.03)
    _tare(ctx)  # hands-free, clear of the object, before the re-approach
    start = ctx.mover._current_ee_pos()
    ctx.mover.approach_to_contact(start, ctx.pose.pos[2], ctx.pose.rpy)  # gentle re-descent
    vac = suction_io.VacuumMonitor()
    vac.start()
    try:
        res = ctx.mover.seal_at_contact(ctx.pose.rpy, vac)  # owns pre-lift + suction on
    finally:
        vac.stop()
    if not res.success:
        logger.warning("[micro] lift cycle {}: re-grab failed ({})", cycle, res.trigger)
    return res.success


def lift_forward(ctx: StepCtx, cycle: int) -> bool:
    """Lift the grabbed object straight up to transport height (recorded).
    Success = the object is still sealed at the top (not dropped mid-lift)."""
    vac = suction_io.VacuumMonitor()
    vac.start()
    try:
        ctx.mover.lift(cfg.SAFE_TRANSPORT_Z)
        held = bool(vac.is_sealed())
    finally:
        vac.stop()
    if not held:
        logger.warning("[micro] lift cycle {}: seal lost during lift", cycle)
    return held


STEPS = {
    "reach_hover": (reach_reset, reach_forward),
    "approach":    (approach_reset, approach_forward),
    "scan_read":   (scan_reset, scan_forward),
    "seal":        (seal_reset, seal_forward),
    "lift":        (lift_reset, lift_forward),
}


def main(
    task: str,
    step: str,
    cycles: int = 20,
    record_dir: str = "recordings_micro",
    instruction: str = "",
    xy_jitter: float = 0.000,
    reach_scatter: float = 0.05,
    dry_run: bool = False,
    home: bool = True,
    seed: int = 0,
    skip_confirmation: bool = False,
) -> bool:
    """Loop one micro-step of one pick sub-task, recording each forward stroke.

    Args:
        task: which pick sub-task to anchor at (see TASKS).
        step: which micro-movement to loop (see STEPS). Use dashes or
            underscores (e.g. ``reach-hover`` or ``reach_hover``).
        cycles: how many forward strokes to record.
        record_dir: output root for takes (separate from the demo's).
        instruction: override the drafted micro-step instruction.
        xy_jitter: per-cycle x,y jitter for approach/seal starts (m).
        reach_scatter: x,y scatter for the reach_hover start (m).
        dry_run: run the motion without recording (eyeball the strokes).
        home: home both arms before starting (like run_demo).
        seed: RNG seed for the jitter/scatter (reproducible runs).
        skip_confirmation: skip the safety prompt.
    """
    step = step.replace("-", "_")
    if task not in TASKS:
        logger.error("unknown task {!r}; choices: {}", task, list(TASKS)); return False
    if step not in STEPS:
        logger.error("unknown step {!r}; choices: {}", step, list(STEPS)); return False
    if step == "scan_read" and not TASKS[task]["scan"]:
        logger.error("step 'scan_read' needs a barcode; {!r} has none.", task); return False

    pose = _taught_pose(TASKS[task]["anchor"])
    place = _taught_pose(TASKS[task]["place_dst"])
    reach_start = np.array([place.pos[0], place.pos[1], cfg.SAFE_TRANSPORT_Z])
    phase = f"{task}.{step}"
    instr = instruction or STEP_INSTR[step].format(thing=TASKS[task]["thing"])
    parent_instr = TASKS[task]["parent"]

    logger.warning("=" * 60)
    logger.warning("Micro-task collection: {} x{} cycles", phase, cycles)
    logger.warning("  instruction : {!r}", instr)
    logger.warning("  anchor      : {} pos={} ", TASKS[task]["anchor"], pose.pos.round(4).tolist())
    logger.warning("  reach start : {} -> {}", TASKS[task]["place_dst"], reach_start.round(4).tolist())
    logger.warning("  out_dir     : {}", os.path.abspath(record_dir) if not dry_run else "(dry-run, not recording)")
    logger.warning("About to move the REAL robot arm with suction. Keep the e-stop reachable.")
    logger.warning("=" * 60)
    if not skip_confirmation and input("Continue? [y/N]: ").strip().lower() != "y":
        logger.info("Cancelled."); return False

    from dexcontrol.core.config import get_robot_config
    robot_configs = get_robot_config()
    robot_configs.enable_sensor("head_camera")
    robot_configs.sensors["head_camera"].transport = "zenoh"

    reset_fn, forward_fn = STEPS[step]

    with Robot(configs=robot_configs) as bot:
        suction_io.suction_off()  # never grab during the approach
        with SuctionMover(bot) as mover:
            if not mover.ensure_ready(release_estop=mover.software_estop_active()):
                logger.error("Arm not ready (E-Stop active?). Aborting."); return False
            if home:
                go_to_default_pose(bot)
            set_head_pitch(bot, angle=30.0)

            recorder = None
            publisher = None
            if not dry_run:
                recorder = RecordController(
                    out_dir=record_dir, spool_dir=DEFAULT_SPOOL_DIR, instruction=instr,
                )
                recorder.set_meta_extra({
                    "parent_task": task,
                    "parent_instruction": parent_instr,
                    "micro_step": step,
                })
                recorder.start()
                publisher = DashboardPublisher(bot, on_sample=recorder.feed).start()

            ctx = StepCtx(
                mover=mover, pose=pose,
                hover_z=float(pose.pos[2]) + cfg.HOVER_HEIGHT_M,
                scan_floor_z=float(pose.pos[2]) + cfg.BCR_SCAN_FLOOR_OFFSET_M,
                reach_start=reach_start,
                rng=np.random.default_rng(seed),
                xy_jitter=xy_jitter, reach_scatter=reach_scatter, recorder=recorder,
            )

            n_ok = 0
            try:
                for cycle in range(cycles):
                    reset_fn(ctx, cycle)                 # reposition — NOT recorded
                    if recorder is not None:
                        recorder.episode_begin(phase)
                        time.sleep(0.2)                  # let the worker flip _recording on
                    success = forward_fn(ctx, cycle)     # scripted stroke — recorded
                    if recorder is not None:
                        recorder.episode_end(success)
                    n_ok += int(success)
                    logger.info("[micro] cycle {}/{}: {} -> {}",
                                cycle + 1, cycles, step, "OK" if success else "FAIL")
            except KeyboardInterrupt:
                logger.warning("[micro] interrupted after {} cycle(s).", cycle)
            finally:
                suction_io.suction_off()
                if publisher is not None:
                    publisher.stop()
                if recorder is not None:
                    recorder.stop()
            logger.info("[micro] done: {}/{} strokes succeeded.", n_ok, cycles)
    return True


if __name__ == "__main__":
    success = tyro.cli(main)
    raise SystemExit(0 if success else 1)
