#!/usr/bin/env python3
"""Calibrate the URDF torque model against the real robot (motion-based).

The URDF does not know the actual end-effector payload (gripper + suction
tool), and the mapping from joint torque [Nm] to measured motor current [A]
is undocumented. On top of that, static-hold currents are dominated by
stiction (run-to-run spread of ~1 A on the shoulder joints), so this script
fits on data collected DURING slow motion, where kinetic friction is
repeatable and is modeled explicitly:

  1. Moves the left arm through the designated calibration poses in
     CALIB_POSE_OFFSETS (forward then reverse, so every joint moves in both
     directions), sampling joint position/velocity/current continuously.
  2. Fits, per joint:   signal_j = k_j * tau_j(q; m) + c_j * sign(v_j) + b_j
     where tau is the URDF gravity torque plus an extra point mass m at
     L_gripper_base (linear-exact basis, m shared across joints, found by
     grid search on variance-normalized SSE so no single joint dominates),
     and c_j * sign(v_j) is Coulomb friction. Only samples where a joint is
     actually moving (|v| > V_EPS) are used — this also excludes any samples
     where a stationary joint might be brake-locked.
  3. Validates with a live wiggle: streams the residual |signal - prediction|
     and reports the free-motion peaks — the noise floor that collision
     thresholds must sit above.

Before any motion, the FK-predicted EE position of every calibration pose is
printed for a final go/no-go confirmation. Use --pose-scale 0.5 for a smaller
first sweep, and edit CALIB_POSE_OFFSETS to designate your own poses.
Start the sweep from a pose away from joint limits (e.g. the default pose) —
poses that clip at a limit lose excitation.

Outputs:
  Collision/calibration_left.json          fitted m, k, c, b + residual stats
  Collision/logs/calib_motion_<ts>.csv     raw motion samples (q, v, y)
  Collision/logs/calib_validate_<ts>.csv   per-tick validation residuals

Usage:
    python calibrate_gravity_model.py                     # full sweep (~2 min)
    python calibrate_gravity_model.py --pose-scale 0.5    # smaller motions
"""

import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np
import pinocchio as pin
import tyro
from loguru import logger

from dexcontrol.robot import Robot

from demo_move_left_ee import CONTROL_DT, LOG_DIR, URDF_PATH, LeftArmIK, check_environment

SAMPLE_DT = 0.02
# Joint velocity deadband [rad/s]: samples where a joint moves slower than this
# are excluded from that joint's fit (stiction regime / possibly brake-locked).
V_EPS = 0.03
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_left.json")

# Designated calibration poses: joint-space offsets [rad] from the start pose,
# clipped to joint limits before execution. Estimating the payload mass and the
# per-joint torque-to-signal gains needs poses where the gravity torque of each
# load-bearing joint takes clearly different values — j2 (shoulder pitch),
# j4 (elbow) and j6 (wrist pitch) dominate, so they are excited individually
# and in combination. Edit this list to designate your own poses.
#   index:            j1     j2     j3     j4     j5     j6     j7
CALIB_POSE_OFFSETS = [
    np.array([0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]),   # start pose
    np.array([+0.30,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]),  # shoulder j1 +
    np.array([-0.30,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]),  # shoulder j1 -
    np.array([0.00, +0.35,  0.00,  0.00,  0.00,  0.00,  0.00]),   # shoulder up
    np.array([0.00, -0.35,  0.00,  0.00,  0.00,  0.00,  0.00]),   # shoulder down
    np.array([0.00,  0.00,  0.00, +0.40,  0.00,  0.00,  0.00]),   # elbow open
    np.array([0.00,  0.00,  0.00, -0.40,  0.00,  0.00,  0.00]),   # elbow closed
    np.array([0.00,  0.00, +0.30,  0.00,  0.00,  0.00,  0.00]),   # shoulder roll
    np.array([0.00,  0.00,  0.00,  0.00,  0.00, +0.40,  0.00]),   # wrist up
    np.array([0.00,  0.00,  0.00,  0.00,  0.00, -0.40,  0.00]),   # wrist down
    np.array([0.00, +0.25,  0.00, -0.30, +0.30, +0.25,  0.00]),   # combo A
    np.array([0.00, -0.25,  0.00, +0.30, -0.30, -0.25,  0.00]),   # combo B
    np.array([0.00, +0.20, +0.20, +0.30,  0.00,  0.00, +0.40]),   # combo C
    # Large-swing poses: decorrelate the arm's own gravity torque from the
    # payload torque so the payload mass m is separable from the gains k
    # (synthetic-recovery std of m: 0.68 kg -> 0.41 kg with these added).
    np.array([0.00, +0.60,  0.00,  0.00,  0.00,  0.00,  0.00]),   # shoulder high
    np.array([0.00,  0.00,  0.00, +0.70,  0.00,  0.00,  0.00]),   # elbow extended
    np.array([0.00,  0.00,  0.00, -0.70,  0.00,  0.00,  0.00]),   # elbow folded
    np.array([0.00, -0.40,  0.00, +0.60,  0.00, +0.40,  0.00]),   # low + extended
    np.array([0.00, +0.40,  0.00, -0.60,  0.00, -0.40,  0.00]),   # high + folded
    np.array([-0.40,  0.00,  0.00, +0.50,  0.00,  0.00,  0.00]),  # j1 swing + elbow
]


# ── model helpers ─────────────────────────────────────────────────────────────

def build_payload_model(ik: LeftArmIK) -> tuple:
    """Return (model, data) equal to the URDF plus 1 kg point mass at L_gripper_base.

    Used as the linear payload basis: gravity(q; m) = gravity_urdf(q) + m * phi(q)
    with phi(q) = gravity_1kg(q) - gravity_urdf(q).
    """
    payload = LeftArmIK()  # fresh URDF load so ik.model stays untouched
    model = payload.model
    fid = model.getFrameId("L_gripper_base")
    frame = model.frames[fid]
    jid = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
    extra = pin.Inertia(1.0, frame.placement.translation.copy(), np.zeros((3, 3)))
    model.inertias[jid] = model.inertias[jid] + extra
    return model, model.createData()


def gravity_basis(
    ik: LeftArmIK, model_p, data_p, data_u, q_full: np.ndarray, vidx: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Left-arm gravity torque of the bare URDF and the +1 kg payload basis."""
    g_u = pin.computeGeneralizedGravity(ik.model, data_u, q_full)[vidx].copy()
    g_p = pin.computeGeneralizedGravity(model_p, data_p, q_full)[vidx].copy()
    return g_u, g_p - g_u


def q_full_from_left(ik: LeftArmIK, left_q: np.ndarray) -> np.ndarray:
    """Full configuration vector with the given left-arm joints substituted in.

    Clipped to model limits: measured joint positions can sit a few 1e-5 rad
    outside the URDF limits, which pink's solve_ik rejects.
    """
    q = ik.configuration.q.copy()
    for j, idx in enumerate(ik.left_idx):
        q[ik.model.idx_qs[idx]] = left_q[j]
    return np.clip(q, ik.model.lowerPositionLimit, ik.model.upperPositionLimit)


def make_pose_targets(ik: LeftArmIK, q_home: np.ndarray, scale: float) -> list[np.ndarray]:
    """Designated calibration poses as absolute left-arm joint targets, limit-clipped."""
    lo = np.array([ik.model.lowerPositionLimit[ik.model.idx_qs[i]] for i in ik.left_idx]) + 0.05
    hi = np.array([ik.model.upperPositionLimit[ik.model.idx_qs[i]] for i in ik.left_idx]) - 0.05
    return [np.clip(q_home + scale * off, lo, hi) for off in CALIB_POSE_OFFSETS]


def preview_pose_targets(ik: LeftArmIK, q_home: np.ndarray, targets: list[np.ndarray]) -> None:
    """Print FK EE position of every designated pose for a go/no-go check."""
    ik.configuration.update(q_full_from_left(ik, q_home))
    pos_home, _ = ik.left_ee_pose()
    logger.info(f"Start EE pos (arm_center frame): {np.round(pos_home, 3)}")
    for i, q_t in enumerate(targets):
        ik.configuration.update(q_full_from_left(ik, q_t))
        pos, _ = ik.left_ee_pose()
        logger.info(
            f"  pose {i:2d}: EE {np.round(pos, 3)}  "
            f"(offset {np.round(pos - pos_home, 3)}, "
            f"max joint delta {np.max(np.abs(q_t - q_home)):.2f} rad)"
        )
    ik.configuration.update(q_full_from_left(ik, q_home))  # restore


# ── fitting ───────────────────────────────────────────────────────────────────

def fit_motion_calibration(
    tau_u: np.ndarray,
    phi: np.ndarray,
    v: np.ndarray,
    y: np.ndarray,
    m_max: float = 3.0,
    m_step: float = 0.05,
    min_excitation: float = 0.05,
    min_samples: int = 30,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit y_j = k_j*(tau_u + m*phi)_j + c_j*sign(v_j) + d_j*v_j + b_j on moving samples.

    Per joint, only samples with |v_j| > V_EPS are used (kinetic-friction
    regime; excludes stiction plateaus and brake-locked joints). The shared
    payload mass m is grid-searched on SSE normalized by each joint's signal
    variance, so a high-current joint cannot single-handedly drag m away.
    The viscous term d*v makes the prediction extrapolate to speeds faster
    than the calibration strokes.

    Joints with too few moving samples or too little torque variation fall
    back to a constant prediction (k=c=d=0, b=mean). Joints that only ever
    move in one direction lose the sign(v) column (it would be collinear
    with the intercept); Coulomb friction is then absorbed into b.

    Args:
        tau_u: URDF gravity torques, shape (S, 7).
        phi: +1 kg payload torque basis, shape (S, 7).
        v: Measured joint velocities, shape (S, 7).
        y: Measured signal (current [A] or torque [Nm]), shape (S, 7).

    Returns:
        (m, k, c, d, b, rms, identifiable) with per-joint arrays.
    """
    sgn = np.where(np.abs(v) > V_EPS, np.sign(v), 0.0)
    masks = [sgn[:, j] != 0 for j in range(7)]
    weights = np.zeros(7)
    for j in range(7):
        if masks[j].sum() >= min_samples:
            var = float(y[masks[j], j].var())
            weights[j] = 1.0 / var if var > 1e-9 else 0.0

    best = None
    for m in np.arange(0.0, m_max + 1e-9, m_step):
        X = tau_u + m * phi
        k = np.zeros(7)
        c = np.zeros(7)
        d = np.zeros(7)
        b = np.zeros(7)
        sse = 0.0
        for j in range(7):
            mj = masks[j]
            if mj.sum() < min_samples or X[mj, j].std() < min_excitation:
                b[j] = float(np.mean(y[:, j]))
                continue  # constant fallback; does not vote on m
            s_j = sgn[mj, j]
            has_both_dirs = bool((s_j > 0).any() and (s_j < 0).any())
            cols = [X[mj, j]] + ([s_j] if has_both_dirs else []) \
                + [v[mj, j], np.ones(mj.sum())]
            A = np.stack(cols, axis=1)
            coef, *_ = np.linalg.lstsq(A, y[mj, j], rcond=None)
            if has_both_dirs:
                k[j], c[j], d[j], b[j] = coef
            else:
                k[j], d[j], b[j] = coef
            r = y[mj, j] - A @ coef
            sse += weights[j] * float(r @ r) / mj.sum()
        if best is None or sse < best[0]:
            best = (sse, float(m), k, c, d, b)
    _, m, k, c, d, b = best

    X = tau_u + m * phi
    pred = k * X + c * sgn + d * v + b
    rms = np.zeros(7)
    identifiable = np.zeros(7, dtype=bool)
    for j in range(7):
        mj = masks[j]
        identifiable[j] = bool(mj.sum() >= min_samples and X[mj, j].std() >= min_excitation)
        sel = mj if identifiable[j] else np.ones(len(y), dtype=bool)
        rms[j] = float(np.sqrt(np.mean((y[sel, j] - pred[sel, j]) ** 2)))
    return m, k, c, d, b, rms, identifiable


def predict_signal(
    k: np.ndarray, c: np.ndarray, d: np.ndarray, b: np.ndarray, m: float,
    g_u: np.ndarray, phi: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """Model-predicted signal: gravity + Coulomb + viscous friction terms."""
    sgn = np.where(np.abs(v) > V_EPS, np.sign(v), 0.0)
    return k * (g_u + m * phi) + c * sgn + d * v + b


# ── robot I/O ─────────────────────────────────────────────────────────────────

def torque_available(arm) -> bool:
    """True if the firmware publishes a joint torque field."""
    try:
        t = arm.get_joint_torque()
        return t is not None and len(t) == 7
    except Exception:
        return False


def read_signal(arm, use_torque: bool) -> np.ndarray:
    return (arm.get_joint_torque() if use_torque else arm.get_joint_current()).astype(float)


def move_left(bot: Robot, q_from: np.ndarray, q_to: np.ndarray, duration: float) -> None:
    """Smoothstep the left arm between two joint configurations."""
    n_steps = max(1, int(duration / CONTROL_DT))
    for step in range(n_steps):
        t = (step + 1) / n_steps
        alpha = t * t * (3 - 2 * t)
        bot.set_joint_pos({"left_arm": q_from + alpha * (q_to - q_from)})
        time.sleep(CONTROL_DT)


def collect_motion_stroke(
    bot: Robot,
    use_torque: bool,
    q_from: np.ndarray,
    q_to: np.ndarray,
    duration: float,
    writer,
    seg: int,
    t0: float,
    store: dict,
) -> None:
    """Smoothstep one stroke while sampling (q, v, signal) every control tick."""
    n_steps = max(1, int(duration / CONTROL_DT))
    for step in range(n_steps):
        t = (step + 1) / n_steps
        alpha = t * t * (3 - 2 * t)
        bot.set_joint_pos({"left_arm": q_from + alpha * (q_to - q_from)})

        q = bot.left_arm.get_joint_pos().astype(float)
        vel = bot.left_arm.get_joint_vel().astype(float)
        y = read_signal(bot.left_arm, use_torque)
        store["q"].append(q)
        store["v"].append(vel)
        store["y"].append(y)
        writer.writerow([f"{time.time() - t0:.3f}", seg]
                        + [f"{x:.5f}" for x in q]
                        + [f"{x:.5f}" for x in vel]
                        + [f"{x:.5f}" for x in y])
        time.sleep(CONTROL_DT)


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    pose_scale: float = 1.0,
    amp_x: float = 0.10,
    amp_y: float = 0.15,
    amp_z: float = 0.10,
    n_validate: int = 8,
    joint_speed: float = 0.5,
    stroke_duration: float = 2.5,
    settle: float = 0.3,
    m_max: float = 3.0,
) -> None:
    """Identify EE payload mass, per-joint gains and friction from motion data.

    Args:
        pose_scale: Scale factor on the designated CALIB_POSE_OFFSETS
            (0.5 = half-size sweep for a cautious first run).
        amp_x: Validation random-stroke range [m] forward/back — match the
            motion your application actually performs (demo defaults).
        amp_y: Validation random-stroke range [m] left/right.
        amp_z: Validation random-stroke range [m] up/down.
        n_validate: Number of random validation strokes.
        joint_speed: Peak joint speed [rad/s] of the validation strokes —
            match your application; thresholds are only valid for the motion
            envelope they were measured on.
        stroke_duration: Seconds per move between calibration poses. Slower
            strokes keep the quasi-static assumption solid; do not go below ~2 s.
        settle: Pause [s] between strokes (not sampled).
        m_max: Upper bound [kg] of the payload mass grid search — keep this at
            a physically plausible value; it acts as a prior. If the fit lands
            exactly on this bound, the model is absorbing a non-gravity effect;
            weigh the tool rather than raising the bound.
    """
    check_environment()

    logger.info("Loading robot model (URDF + payload basis)...")
    ik = LeftArmIK()
    model_p, data_p = build_payload_model(ik)
    data_u = ik.model.createData()
    vidx = [ik.model.idx_vs[idx] for idx in ik.left_idx]

    n_strokes = 2 * (len(CALIB_POSE_OFFSETS) - 1)
    logger.warning("=" * 60)
    logger.warning(f"The LEFT arm will sweep {len(CALIB_POSE_OFFSETS)} designated poses")
    logger.warning(f"forward and back ({n_strokes} strokes, ~{n_strokes * (stroke_duration + settle):.0f} s, "
                   f"scale {pose_scale:.2f}), then do one validation wiggle.")
    logger.warning("Clear the arm's workspace; keep the e-stop within reach.")
    logger.warning("=" * 60)
    if input("Connect to robot and preview poses? [y/N]: ").lower() != "y":
        logger.info("Cancelled by user")
        sys.exit(0)

    logger.info(f"Connecting to robot: {os.environ['ROBOT_NAME']}")
    bot = Robot()

    try:
        use_torque = torque_available(bot.left_arm)
        signal_name = "torque [Nm]" if use_torque else "current [A]"
        logger.info(f"Firmware torque readback: {'YES — fitting on torque' if use_torque else 'no — fitting on current'}")
        try:
            brake = bot.left_arm.get_brake_status()
            logger.info(f"Brake status: enabled={brake.get('enabled')} joints={brake.get('joints')}")
        except Exception as exc:
            logger.info(f"Brake status query unavailable ({exc}) — the moving-sample filter covers this.")

        ik.sync_from_robot(bot)
        q_home = bot.left_arm.get_joint_pos().astype(float)
        pose_targets = make_pose_targets(ik, q_home, pose_scale)

        # FK preview of every designated pose, then final go/no-go
        preview_pose_targets(ik, q_home, pose_targets)
        if input("Execute these poses? [y/N]: ").lower() != "y":
            logger.info("Cancelled by user")
            return

        # ── Phase 1: motion sweep (forward then reverse) ──────────────────────
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        motion_path = os.path.join(LOG_DIR, f"calib_motion_{ts}.csv")
        torso_q = bot.torso.get_joint_pos().astype(float)
        store: dict = {"q": [], "v": [], "y": []}
        # Forward through the designated poses, then back again so every joint
        # collects samples in both velocity directions (needed for the sign(v) term).
        sequence = pose_targets[1:] + pose_targets[-2::-1]
        with open(motion_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "seg"]
                            + [f"q{j+1}" for j in range(7)]
                            + [f"v{j+1}" for j in range(7)]
                            + [f"y{j+1}" for j in range(7)])
            # Torso pose (constant during the sweep) so the gravity basis can be
            # rebuilt offline: stored as a trailer row with seg="torso".
            t0 = time.time()
            q_prev = q_home
            for i, q_target in enumerate(sequence):
                logger.info(f"Stroke {i+1}/{len(sequence)}")
                collect_motion_stroke(
                    bot, use_torque, q_prev, q_target, stroke_duration,
                    writer, i, t0, store,
                )
                q_prev = q_target
                time.sleep(settle)
            writer.writerow(["torso", ""] + [f"{x:.5f}" for x in torso_q] + [""] * 11)
        logger.info(f"Motion samples saved to {motion_path}  ({len(store['q'])} ticks)")

        logger.info("Returning to start pose...")
        move_left(bot, q_prev, q_home, stroke_duration)
        time.sleep(settle)

        # ── Phase 2: fit ──────────────────────────────────────────────────────
        q_arr = np.asarray(store["q"])
        v_arr = np.asarray(store["v"])
        y_arr = np.asarray(store["y"])
        logger.info(f"Computing gravity basis for {len(q_arr)} samples...")
        tau_u = np.zeros((len(q_arr), 7))
        phi = np.zeros((len(q_arr), 7))
        for i in range(len(q_arr)):
            tau_u[i], phi[i] = gravity_basis(ik, model_p, data_p, data_u, q_full_from_left(ik, q_arr[i]), vidx)

        m, k, c, d, b, rms, identifiable = fit_motion_calibration(tau_u, phi, v_arr, y_arr, m_max=m_max)
        logger.info("=" * 60)
        logger.info(f"Fitted EE payload mass:  {m:.2f} kg (point mass at L_gripper_base)")
        if m >= m_max - 1e-9:
            logger.warning(
                f"Payload mass hit the search bound ({m_max} kg) — treat m as a capped "
                f"effective parameter, not a physical mass. If residuals are acceptable "
                f"this is fine for collision detection; otherwise weigh the tool and fix m."
            )
        logger.info(f"Per-joint gain k [{signal_name} per Nm]:   {np.round(k, 4)}")
        logger.info(f"Per-joint Coulomb friction c:              {np.round(c, 4)}")
        logger.info(f"Per-joint viscous friction d [per rad/s]:  {np.round(d, 4)}")
        logger.info(f"Per-joint offset b:                        {np.round(b, 4)}")
        logger.info(f"Moving-sample fit RMS residual:            {np.round(rms, 4)}")
        if not np.all(identifiable):
            bad = np.where(~identifiable)[0] + 1
            logger.warning(
                f"Joints {bad.tolist()} had too few moving samples or ~no gravity "
                f"torque variation — using constant baseline for them (k=c=0)."
            )

        # ── Phase 3: validation with random 3-D strokes ───────────────────────
        # Random targets matching the application's motion envelope (ranges and
        # speed), so the measured noise floor — and the thresholds derived from
        # it — reflect the motion the monitor will actually supervise.
        logger.info(f"Validation: {n_validate} random strokes within "
                    f"±{amp_x*100:.0f}/±{amp_y*100:.0f}/±{amp_z*100:.0f} cm "
                    f"at {joint_speed:.2f} rad/s peak...")
        ik.configuration.update(q_full_from_left(ik, q_home))
        pos0, rot0 = ik.left_ee_pose()
        rng = np.random.default_rng(0)
        strokes = []
        for i in range(n_validate):
            offset = rng.uniform(-1.0, 1.0, 3) * np.array([amp_x, amp_y, amp_z])
            strokes.append((f"rand{i}", ik.solve_left(pos0 + offset, rot0)))
        strokes.append(("home", q_home))

        validate_path = os.path.join(LOG_DIR, f"calib_validate_{ts}.csv")
        max_res = np.zeros(7)
        max_banded = np.zeros(7)
        # Change-layer (impact detector) floor: residual change over a 0.1 s
        # window, tolerating ±c when the window crosses a stop/reversal.
        change_window_s = 0.1
        hist: deque = deque(maxlen=int(round(change_window_s / CONTROL_DT)) + 1)
        max_change = np.zeros(7)
        with open(validate_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "phase"]
                            + [f"q{j+1}" for j in range(7)]
                            + [f"v{j+1}" for j in range(7)]
                            + [f"y{j+1}" for j in range(7)]
                            + [f"pred{j+1}" for j in range(7)]
                            + [f"res{j+1}" for j in range(7)])
            t0 = time.time()
            q_prev = q_home
            for label, q_target in strokes:
                # Constant peak speed: smoothstep peak velocity = 1.5 * dq / T
                duration = float(np.clip(
                    1.5 * np.max(np.abs(q_target - q_prev)) / joint_speed, 0.8, 6.0))
                n_steps = max(1, int(duration / CONTROL_DT))
                for step in range(n_steps):
                    t = (step + 1) / n_steps
                    alpha = t * t * (3 - 2 * t)
                    bot.set_joint_pos({"left_arm": q_prev + alpha * (q_target - q_prev)})

                    q_meas = bot.left_arm.get_joint_pos().astype(float)
                    v_meas = bot.left_arm.get_joint_vel().astype(float)
                    y_meas = read_signal(bot.left_arm, use_torque)
                    g_u, ph = gravity_basis(ik, model_p, data_p, data_u, q_full_from_left(ik, q_meas), vidx)
                    pred = predict_signal(k, c, d, b, m, g_u, ph, v_meas)
                    res = np.abs(y_meas - pred)
                    max_res = np.maximum(max_res, res)
                    # Stiction-band-aware residual: a stopped joint may sit
                    # anywhere within ±c of the gravity prediction, so only
                    # the part outside that band counts (what the monitor uses).
                    band = np.where(np.abs(v_meas) <= V_EPS, np.abs(c), 0.0)
                    max_banded = np.maximum(max_banded, np.maximum(0.0, res - band))
                    res_signed = y_meas - pred
                    if len(hist) == hist.maxlen:
                        res_old, v_old = hist[0]
                        moving_both = (np.abs(v_meas) > V_EPS) & (np.abs(v_old) > V_EPS)
                        band_c = np.where(moving_both, 0.0, np.abs(c))
                        max_change = np.maximum(
                            max_change,
                            np.maximum(0.0, np.abs(res_signed - res_old) - band_c),
                        )
                    hist.append((res_signed, v_meas))

                    writer.writerow([f"{time.time() - t0:.3f}", label]
                                    + [f"{x:.5f}" for x in q_meas]
                                    + [f"{x:.5f}" for x in v_meas]
                                    + [f"{x:.5f}" for x in y_meas]
                                    + [f"{x:.5f}" for x in pred]
                                    + [f"{x:.5f}" for x in res])
                    time.sleep(CONTROL_DT)
                q_prev = q_target

        logger.info(f"Free-motion max residual per joint:        {np.round(max_res, 4)}")
        logger.info(f"Free-motion max BANDED residual per joint: {np.round(max_banded, 4)}")
        suggested = np.round(1.5 * max_res, 4)
        # Banded thresholds: 1.5x the banded free-motion peak, floored at 3x the
        # fit noise so a joint that never spiked doesn't get a hair-trigger.
        banded_suggested = np.round(np.maximum(1.5 * max_banded, 3.0 * rms), 4)
        change_suggested = np.round(np.maximum(1.5 * max_change, 3.0 * rms), 4)
        logger.info(f"Suggested banded collision thresholds:     {banded_suggested}  ({signal_name})")
        logger.info(f"Suggested change-layer thresholds:          {change_suggested}  ({signal_name})")

        calib = {
            "fit": "motion",
            "signal": "torque" if use_torque else "current",
            "v_eps": V_EPS,
            "payload_mass_kg": m,
            "k": k.tolist(),
            "c": c.tolist(),
            "d": d.tolist(),
            "b": b.tolist(),
            "moving_rms": rms.tolist(),
            "identifiable": identifiable.tolist(),
            "free_motion_max_residual": max_res.tolist(),
            "banded_free_motion_max_residual": max_banded.tolist(),
            "suggested_thresholds": suggested.tolist(),
            "banded_suggested_thresholds": banded_suggested.tolist(),
            "change_window_s": change_window_s,
            "change_free_motion_max_residual": max_change.tolist(),
            "change_suggested_thresholds": change_suggested.tolist(),
            "pose_scale": pose_scale,
            "validation_amps": [amp_x, amp_y, amp_z],
            "validation_joint_speed": joint_speed,
            "urdf": URDF_PATH,
            "timestamp": ts,
        }
        with open(CALIB_PATH, "w") as f:
            json.dump(calib, f, indent=2)
        logger.info(f"Calibration saved to {CALIB_PATH}")
        logger.info(f"Validation log:      {validate_path}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        logger.info("Shutting down; arm holds its last commanded position.")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
