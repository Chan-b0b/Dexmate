"""Manual EE jog — print the live torso / EE pose, then move to a typed x y z.

For taking EE positions by hand (e.g. the lid drop-off spot): the cup is held
VERTICAL the whole time (roll/pitch from cfg.GRASP_ORIENTATION_RPY, a fixed
yaw), the leg is a straight Cartesian line streamed per tick at creep speed,
and every target is IK-checked before anything moves. Left arm by default.

The ORIENTATION is taken as it is too: the hold is whatever the arm starts in,
so stepping always works from any stance (`v` levels the cup where that is
reachable, --vertical demands it from the start).

The torso is taken AS IT IS — the model is built from the live torso and the
torso is pinned there (unlike the demo, which refuses any stance but
cfg.TORSO_JOINTS). ``t`` moves it and rebuilds the model at the new stance, so
a stance can be found and its EE positions taken in one session. Note every
base_link EE number printed here is only valid at the torso stance printed
next to it.

    python -m LGES.ik_demo.jog_ee
    python -m LGES.ik_demo.jog_ee --speed 0.02       # slower than the default creep
    python -m LGES.ik_demo.jog_ee --yaw-deg 90       # different wrist yaw
    python -m LGES.ik_demo.jog_ee --free-yaw         # let the solver pick the yaw
    python -m LGES.ik_demo.jog_ee --torso-vel 0.1    # gentler torso moves

Single keypresses, no Enter (base_link axes, one --step per press):

    w / s   +x / -x   (forward / back)      r / f   +z / -z   (up / down)
    a / d   +y / -y   (left / right)        , / .   yaw -/+ (--yaw-step)
    - / +   step /2 or x2                   v       level the cup (vertical)
    p       print the full pose
    g       type an absolute `x y z`        t       type torso `j1 j2 j3` (deg)
    q / ESC quit (leaves the arm where it is)

Every press is IK-checked before it moves, so a blocked direction just prints
`blocked` and the arm stays put. Steps are taken from the last COMMANDED pose,
not the measured one, so repeated presses do not accumulate the tracking
error. A non-tty stdin falls back to the same keys typed + Enter.

E-stop in reach: this moves the real arm.
"""

from __future__ import annotations

import argparse
import sys
import termios
import time
import tty

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from . import config as cfg
from .arm import ArmMover, TickPacer, connect_robot, move_torso


def _fmt(p) -> str:
    return "(" + ", ".join(f"{float(v):+.4f}" for v in p) + ")"


def _report(m: ArmMover, rpy) -> None:
    torso = np.asarray(m._robot.torso.get_joint_pos(), dtype=float)
    pos, live_rpy = m.current_ee_pose()
    logger.info("torso  {} rad = {} deg", np.round(torso, 4), np.round(np.rad2deg(torso), 1))
    logger.info("ee pos {}  rpy {} deg", _fmt(pos), _fmt(np.rad2deg(live_rpy)))
    logger.info("hold   rpy {} deg (vertical cup)", _fmt(np.rad2deg(rpy)))


def _torso_go(m: ArmMover, target, vel_scale: float, timeout: float) -> None:
    """Torso motion at ``vel_scale`` of its velocity ceiling, blocking until it
    arrives. arm.move_torso picks the right call for the installed dexcontrol
    (see it for why there are two); this keeps its own scale so a jog session
    can be slower than the demo's cfg.TORSO_VEL_SCALE."""
    state = move_torso(m._robot.torso, target, vel_scale, timeout)
    if state != "finished":
        logger.warning("[jog] torso motion ended '{}' ({}s timeout)", state, timeout)


def _move_torso(m: ArmMover, deg, vel_scale: float) -> None:
    """Command the torso to ``deg``, then rebuild the arm model at the stance
    it actually reached (the reduced model bakes the torso in — every IK solve
    and reach verdict after this would otherwise be answered at the OLD
    stance). The arm joints are untouched, so the last commanded arm config
    stays valid; the EE moves in base_link because its base did."""
    target = np.deg2rad(np.asarray(deg, dtype=float))
    live = np.asarray(m._robot.torso.get_joint_pos(), dtype=float)
    logger.warning("[jog] torso {} -> {} deg — the whole arm swings with it",
                   np.round(np.rad2deg(live), 1), np.round(np.rad2deg(target), 1))
    logger.warning("[jog] = {} rad (paste into cfg.TORSO_JOINTS)", np.round(target, 6))
    if input(f"Move the torso at velocity_scale={vel_scale}? [y/N]: ").strip().lower() != "y":
        logger.info("[jog] torso move cancelled")
        return
    _torso_go(m, target, vel_scale, timeout=60.0)
    reached = np.asarray(m._robot.torso.get_joint_pos(), dtype=float)
    d = float(np.max(np.abs(reached - target)))
    if d > 0.02:
        logger.warning("[jog] torso stopped {:.3f} rad ({:.1f} deg) from the target "
                       "(limit / obstruction?) — re-modelling at where it IS",
                       d, float(np.rad2deg(d)))
    m._setup_model(reached)
    m._setup_ik()
    m._setup_collision()


def _jog_line(m: ArmMover, target, rpy, speed: float) -> bool:
    """Straight Cartesian line to ``target`` at ``speed``, smoothstep in/out.

    Same per-tick warm-IK stream as the descent legs (so the commanded path is
    a straight line by construction), just slow end to end — no fast cruise to
    shed tracking error out of. Returns False if a tick's IK falls beyond
    REACH_TOL_M (halts in place, partial motion) or the leg times out.
    """
    dt = 1.0 / float(cfg.CONTROL_HZ)
    prev_q = m._start_q()
    p_now = np.asarray(m.fk(prev_q)[0], dtype=float)
    p_goal = np.asarray(target, dtype=float)
    total = float(np.linalg.norm(p_goal - p_now))
    if total < 1e-4:
        logger.info("[jog] already there ({:.2f}mm) — not moving", total * 1000)
        return True
    ramp = max(float(cfg.DESCENT_RAMP_S), 1e-6)
    band = max(speed * ramp, 1e-6)          # decel distance = the ramp-in distance
    logger.info("[jog] {} -> {}  ({:.1f}mm at {:.3f} m/s)",
                _fmt(p_now), _fmt(p_goal), total * 1000, speed)

    deadline = time.perf_counter() + total / speed + 4.0 * ramp + 3.0
    elapsed = 0.0
    trace_s = float(cfg.DESCENT_TRACE_S)
    trace_t, tick_acc, tick_n, t_tick = 0.0, 0.0, 0, time.perf_counter()
    pace = TickPacer(dt)
    while True:
        d = p_goal - p_now
        dist = float(np.linalg.norm(d))
        if dist <= 5e-4:
            break
        if time.perf_counter() > deadline:
            m._send(prev_q, np.zeros(len(prev_q)))
            logger.warning("[jog] timed out {:.1f}mm short at {} — halting",
                           dist * 1000, _fmt(p_now))
            return False
        r_in = min(1.0, elapsed / ramp)
        r_out = min(1.0, dist / band)
        f = min(r_in, r_out)
        v = speed * max(f * f * (3.0 - 2.0 * f), 0.05)   # floor: smoothstep -> 0 never arrives
        p_next = p_now + d / dist * min(v * dt, dist)
        sol, p_sched = m.solve_step(prev_q, p_now, p_next, rpy, dt)
        if sol.pos_err_m > cfg.REACH_TOL_M:
            m._send(prev_q, np.zeros(len(prev_q)))
            logger.warning("[jog] stalled {:.1f}mm short at {} — halting",
                           sol.pos_err_m * 1000, _fmt(p_now))
            return False
        m._send(sol.q, (sol.q - prev_q) / dt)
        p_now, prev_q = np.asarray(p_sched, dtype=float), sol.q
        elapsed += dt
        pace.wait()
        now = time.perf_counter()
        tick_acc += now - t_tick; tick_n += 1; trace_t += now - t_tick
        t_tick = now
        if trace_s and trace_t >= trace_s:
            m._track_trace("jog", p_now, prev_q, tick_acc / tick_n * 1000.0)
            trace_t, tick_acc, tick_n = 0.0, 0.0, 0
    m._send(prev_q, np.zeros(len(prev_q)))
    return True


def _yaw_candidates(yaw0: float):
    """Wrist yaws to try, nearest-first: the held one, its 180-deg flip (same
    approach for a symmetric part), then outward. Same list run_lid's pick
    search uses, so a yaw found here is one the demo can also fly."""
    out = [0.0, 180.0]
    for d in (10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        out += [d, 180.0 + d, -d, 180.0 - d]
    return [float((yaw0 + np.deg2rad(d) + np.pi) % (2.0 * np.pi) - np.pi) for d in out]


def _resolve(m: ArmMover, target, rpy, free_yaw: bool):
    """(rpy, solution) that reaches ``target``, or (None, best) if none does.
    With ``free_yaw`` the wrist yaw is a free variable — the cup is round, so
    only reachability picks it, and the chosen value is logged because that is
    the number that goes into a config."""
    cands = _yaw_candidates(rpy[2]) if free_yaw else [rpy[2]]
    best = None
    for k, yaw in enumerate(cands):
        r = (rpy[0], rpy[1], yaw)
        # TWO seeds per yaw: min-motion from where the arm is, then a fresh
        # solve from home. A min-motion solve from the live config strands on
        # whatever elbow branch the current pose sits on (49-113mm short on
        # targets that DO solve from another branch — the same retry
        # _view_park needs), and a pose-finding tool must not call those
        # unreachable.
        for seed, tag in ((m._start_q(), "live"), (m._home_seed, "home-seed")):
            sol = m.solve_pose(target, r, seed=seed, min_motion=(tag == "live"))
            if best is None or sol.pos_err_m < best[1].pos_err_m:
                best = (r, sol)
            if (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits
                    and not sol.in_collision):
                if k:
                    logger.warning("held yaw {:+.1f}deg cannot reach it — using "
                                   "{:+.1f}deg ({:+.1f}deg away)",
                                   float(np.rad2deg(rpy[2])), float(np.rad2deg(yaw)),
                                   float(np.rad2deg((yaw - rpy[2] + np.pi)
                                                    % (2 * np.pi) - np.pi)))
                if tag != "live":
                    logger.info("solved via the {} branch", tag)
                return r, sol
    return None, best[1]


def _reorient(m: ArmMover, rpy) -> bool:
    """Bring the cup to ``rpy`` IN PLACE, before a straight-line leg.

    _jog_line holds one rpy from its FIRST tick, so a leg that also has to
    re-orient demands a big joint step for almost no Cartesian motion and that
    tick's IK falls outside REACH_TOL — the leg halts having never moved. 0904
    on the robot: from the joint park (cup at roll -175, pitch +18.5, yaw -160)
    with the usual vertical hold (180, 0, 180), tick one came back 18.4mm short
    and the jog stalled in place. It is the WHOLE orientation, not just the yaw
    — that start is 27 deg away, 18.5 of it in pitch.

    A planned (ruckig) joint move at the current xyz does the rotation
    properly, and the line that follows is then pure translation. Tried from
    the live branch and then from home. False = that orientation is not
    holdable at this xy (offline: the joint park's own xy is 19.2mm short of
    holding the cup vertical, from either branch), and the caller falls back to
    one planned joint move to the target."""
    pos, cur = m.current_ee_pose()
    d = float((Rotation.from_euler("xyz", np.asarray(cur, dtype=float)).inv()
               * Rotation.from_euler("xyz", np.asarray(rpy, dtype=float))).magnitude())
    if d < np.deg2rad(0.5):
        return True
    sol = None
    for seed, tag in ((m._start_q(), "live"), (m._home_seed, "home-seed")):
        sol = m.solve_pose(tuple(pos), tuple(rpy), seed=seed, min_motion=(tag == "live"))
        if (sol.pos_err_m <= cfg.REACH_TOL_M and sol.in_limits
                and not sol.in_collision):
            logger.info("re-orienting {:.1f}deg in place first: rpy {} -> {} "
                        "({} branch)", float(np.rad2deg(d)), _fmt(np.rad2deg(cur)),
                        _fmt(np.rad2deg(rpy)), tag)
            m.move_joints(sol.q)
            return True
    logger.warning("cannot hold rpy {} at the current xy ({:.1f}deg away, best "
                   "{:.1f}mm short)", _fmt(np.rad2deg(rpy)), float(np.rad2deg(d)),
                   sol.pos_err_m * 1000)
    return False


def _getch() -> str:
    """One keypress, no Enter. Falls back to a typed line when stdin is not a
    tty (piped input, some IDE consoles) — the same keys work, they just need
    Enter."""
    if not sys.stdin.isatty():
        return (sys.stdin.readline().strip() or "\n")[:1]
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


_KEYS = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1),
         "r": (2, +1), "f": (2, -1)}


def _brief(m: ArmMover, rpy, step: float) -> None:
    pos, _ = m.fk(m._start_q())
    logger.info("ee {}  yaw {:+.1f}deg   step {:.0f}mm", _fmt(pos),
                float(np.rad2deg(rpy[2])), step * 1000.0)


def _step_to(m: ArmMover, target, rpy, args) -> bool:
    """One relative step: resolve, level the cup if needed, jog the line.
    False = nothing moved (the caller can put its hold back)."""
    use_rpy, sol = _resolve(m, target, rpy, bool(args.free_yaw))
    if use_rpy is None:
        logger.warning("blocked: {} is {:.1f}mm short (in_limits={} collision={})",
                       _fmt(target), sol.pos_err_m * 1000, sol.in_limits,
                       sol.in_collision)
        return False
    if _reorient(m, use_rpy):
        _jog_line(m, target, use_rpy, float(args.speed))
    else:
        logger.warning("moving as a PLANNED JOINT move — the path is NOT a straight "
                       "line, watch it")
        m.move_joints(sol.q)
    return True


def _parse_xyz(text: str):
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        return tuple(float(v) for v in parts)
    except ValueError:
        return None


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--speed", type=float, default=cfg.DESCENT_CREEP_SPEED_M_S,
                    help=f"jog speed m/s (default {cfg.DESCENT_CREEP_SPEED_M_S})")
    ap.add_argument("--vertical", action="store_true",
                    help="force the cup vertical (roll/pitch from "
                         "GRASP_ORIENTATION_RPY) instead of holding the orientation "
                         "the arm starts in — only works where that is reachable")
    ap.add_argument("--yaw-deg", type=float, default=float(np.rad2deg(cfg.GRASP_YAW)),
                    help="wrist yaw for --vertical (default = the demo's grasp yaw)")
    ap.add_argument("--step", type=float, default=0.01,
                    help="metres per keypress (default 0.01; -/+ halves/doubles live)")
    ap.add_argument("--yaw-step", type=float, default=5.0,
                    help="degrees per , / . press (default 5)")
    ap.add_argument("--free-yaw", action="store_true",
                    help="if the held yaw cannot reach a target, search wrist yaws "
                         "(canonical first, then outward) and use the first that "
                         "solves — for finding a pose whose yaw you do not know yet")
    ap.add_argument("--torso-vel", type=float, default=0.2,
                    help="torso velocity_scale in (0,1] for the `t` command (default 0.2)")
    args = ap.parse_args()

    logger.warning("EE jog ({} arm) — moves the real arm at {:.3f} m/s.", args.side, args.speed)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        return

    with connect_robot() as bot:
        m = ArmMover(robot=bot, side=args.side)
        release = m.software_estop_active()
        if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
            return
        if not m.ensure_ready_live_torso(release_estop=release,
                                         vel_scale=float(args.torso_vel)):
            logger.error("arm not ready — aborting")
            return
        # Hold the orientation the arm is ALREADY in, unless --vertical. Forcing
        # the vertical cup by default made the tool unusable from the stance a
        # run leaves the arm in: the joint park has the cup at pitch +18.8 deg
        # and its own xy cannot hold it vertical at all (offline: 15-28mm short
        # whichever way you step), so every key came back `blocked`. `v` levels
        # it wherever that IS reachable.
        _, live_rpy = m.current_ee_pose()
        rpy = ((float(cfg.GRASP_ORIENTATION_RPY[0]), float(cfg.GRASP_ORIENTATION_RPY[1]),
                float(np.deg2rad(args.yaw_deg))) if args.vertical
               else tuple(float(v) for v in live_rpy))
        _report(m, rpy)
        step, yaw_step = float(args.step), float(np.deg2rad(args.yaw_step))
        logger.info("keys: w/s=+-x  a/d=+-y  r/f=+-z  ,/.=yaw  -/+=step  v=level cup  "
                    "p=pose  g=goto  t=torso  q=quit   (step {:.0f}mm)", step * 1000)
        while True:
            try:
                k = _getch()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if k in ("q", "\x1b", "\x03", ""):        # q / ESC / ^C / EOF
                break
            if k in _KEYS:
                axis, sign = _KEYS[k]
                pos, _ = m.fk(m._start_q())          # commanded, not measured
                target = [float(v) for v in pos]
                target[axis] += sign * step
                _step_to(m, tuple(target), rpy, args)
                _brief(m, rpy, step)
            elif k in (",", "."):
                want = (rpy[0], rpy[1],
                        float(rpy[2] + (yaw_step if k == "." else -yaw_step)))
                pos, _ = m.fk(m._start_q())
                if _step_to(m, tuple(float(v) for v in pos), want, args):
                    rpy = want
                _brief(m, rpy, step)
            elif k in ("-", "_"):
                step = max(0.0005, step / 2.0)
                logger.info("step {:.1f}mm", step * 1000)
            elif k in ("+", "="):
                step = min(0.20, step * 2.0)
                logger.info("step {:.1f}mm", step * 1000)
            elif k == "v":
                want = (float(cfg.GRASP_ORIENTATION_RPY[0]),
                        float(cfg.GRASP_ORIENTATION_RPY[1]), float(rpy[2]))
                logger.info("levelling the cup: hold rpy -> {}", _fmt(np.rad2deg(want)))
                if _step_to(m, tuple(float(v) for v in m.fk(m._start_q())[0]),
                            want, args):
                    rpy = want          # only adopt a hold the arm can actually keep
                else:
                    logger.warning("cup NOT levelled here — hold stays at {}",
                                   _fmt(np.rad2deg(rpy)))
                _report(m, rpy)
            elif k == "p":
                _report(m, rpy)
            elif k == "g":
                target = _parse_xyz(input("goto x y z: "))
                if target is None:
                    logger.warning("need three numbers, e.g. 0.90 0.0 0.75")
                else:
                    _step_to(m, target, rpy, args)
                    _report(m, rpy)
            elif k == "t":
                deg = _parse_xyz(input("torso j1 j2 j3 (deg): "))
                if deg is None:
                    logger.warning("torso needs three angles in deg, e.g. 30 110 15")
                else:
                    _move_torso(m, deg, float(args.torso_vel))
                    _report(m, rpy)
    logger.info("done")


if __name__ == "__main__":
    _main()
