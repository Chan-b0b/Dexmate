"""Wrench frame probe: hand-push the part, learn the sensor->base axis mapping.

READ ONLY — no motion is ever commanded.

Why this exists: a corner-descent trace CANNOT settle the sensor->base mapping.
A rotation preserves internal consistency, so force and moment agree with each
other whether the mapping is right, sign-flipped, or swapped 90 deg — only a
push in a KNOWN base direction pins it down. And a sign array
(cfg.WRENCH_AXIS_SIGN) can only express the 8 diagonal corrections: it cannot
fix an axis SWAP, and it cannot fix a reflection unless its own determinant is
negative. Guessing signs run-to-run gives a different answer every run.

    python -m LGES.ik_demo.wrench_probe            # live monitor
    python -m LGES.ik_demo.wrench_probe --fit       # guided 4-push fit

Base frame: +x = forward (away from the robot), +y = LEFT, +z = up.

--fit walks +x / -x / +y / -y, captures each push, and scores all 48 signed
permutations of the sensor axes against the labelled directions. Opposite pairs
are DIFFERENCED ((push+ - push-)/2) so any constant offset — tare drift, a
gravity change from the held part — cancels instead of biasing the fit, and R is
read live per capture rather than reconstructed. Push hard (15N+), hold steady,
and keep the direction as axis-aligned as you can: the discrete 48-way choice
tolerates maybe +/-20 deg of aim error, and the reported residual tells you
whether your pushes were clean enough to trust the answer.
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np
from loguru import logger

from . import config as cfg
from .arm import ArmMover

# (label, base-frame unit direction the operator pushes the PART toward)
_PUSHES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("+x  (FORWARD, away from the robot)", (1.0, 0.0, 0.0)),
    ("-x  (BACKWARD, toward the robot)", (-1.0, 0.0, 0.0)),
    ("+y  (LEFT)", (0.0, 1.0, 0.0)),
    ("-y  (RIGHT)", (0.0, -1.0, 0.0)),
    ("-z  (press DOWN on the part)", (0.0, 0.0, -1.0)),
)


def _read6(w) -> np.ndarray:
    """One raw 6-vector (force, torque); pads torques for force-only sensors."""
    s = np.asarray(w.get_wrench_state(), dtype=float).ravel()
    if s.size < 6:
        s = np.concatenate([s[:3], np.zeros(3)])
    return s[:6]


def _mean(w, n: int, dt: float = 0.005) -> np.ndarray:
    rows = [_read6(w) for _ in range(n) if not time.sleep(dt)]
    return np.mean(rows, axis=0)


def _signed_perms():
    """All 48 signed permutations M: corrected[i] = sign[i] * raw[perm[i]]."""
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = sg[row]
            yield M, perm, sg


def _ang_deg(v: np.ndarray, target: np.ndarray) -> float:
    """Angle between two 3-vectors, degrees (0 = the mapping points it right)."""
    nv, nt = np.linalg.norm(v), np.linalg.norm(target)
    if nv < 1e-9 or nt < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v, target) / (nv * nt), -1.0, 1.0))))


def _live(arm: ArmMover, w) -> None:
    input("[probe] hands OFF the part — Enter to tare: ")
    rows = [_read6(w) for _ in range(200) if not time.sleep(0.005)]
    base = np.mean(rows, axis=0)
    noise = np.std([r[:3] for r in rows], axis=0)
    logger.info("[probe] tared; sensor noise sigma=({:.2f},{:.2f},{:.2f})N", *noise)
    logger.info("[probe] cfg.WRENCH_AXIS_SIGN = {}", cfg.WRENCH_AXIS_SIGN)
    logger.info("[probe] push along ONE base axis (+x = forward, +y = LEFT); Ctrl-C to stop")
    sign = np.asarray(cfg.WRENCH_AXIS_SIGN, dtype=float)
    try:
        while True:
            s = _read6(w)
            f_raw, m_raw = s[:3] - base[:3], s[3:6] - base[3:6]
            R = arm.current_ee_rotation()
            f_now, f_nosign, m_now = R @ (f_raw * sign), R @ f_raw, R @ (m_raw * sign)
            print(f"raw=({f_raw[0]:+6.1f},{f_raw[1]:+6.1f},{f_raw[2]:+6.1f})N  "
                  f"base_now=({f_now[0]:+6.1f},{f_now[1]:+6.1f},{f_now[2]:+6.1f})N  "
                  f"base_raw=({f_nosign[0]:+6.1f},{f_nosign[1]:+6.1f},{f_nosign[2]:+6.1f})N  "
                  f"m_now=({m_now[0]:+5.2f},{m_now[1]:+5.2f})Nm  "
                  f"|f|={np.linalg.norm(f_now):5.1f}N  "
                  f"dominant={'xyz'[int(np.argmax(np.abs(f_now)))]}", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()


def _fit(arm: ArmMover, w) -> None:
    input("[probe] hands OFF the part — Enter to tare: ")
    base = _mean(w, 200)
    logger.info("[probe] tared. Push the PART, hold steady, then press Enter. "
                "Push hard (15N+) and stay axis-aligned.")

    caps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, direction in _PUSHES:
        input(f"\n[probe] push {label} and HOLD — Enter to capture: ")
        raw = _mean(w, 60)[:3] - base[:3]
        R = arm.current_ee_rotation()
        mag = float(np.linalg.norm(raw))
        logger.info("[probe]   raw=({:+.1f},{:+.1f},{:+.1f})N |f|={:.1f}N",
                    *raw, mag)
        if mag < 5.0:
            logger.warning("[probe]   only {:.1f}N — push harder, that push is "
                           "mostly noise", mag)
        caps[label[:2]] = (raw, R, np.asarray(direction, dtype=float))

    # Differential pairs: a constant offset (tare drift, held-part gravity)
    # cancels, leaving the pure push force. z has no opposite push (pulling up
    # would break the seal), so its offset survives — treat z as the weaker row.
    obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for plus, minus in (("+x", "-x"), ("+y", "-y")):
        rp, R, d = caps[plus]
        rm, _, _ = caps[minus]
        diff = (rp - rm) / 2.0
        obs.append((diff, R, d))
        logger.info("[probe] {}/{} differential raw=({:+.1f},{:+.1f},{:+.1f})N "
                    "|f|={:.1f}N", plus, minus, *diff, float(np.linalg.norm(diff)))
    obs.append(caps["-z"])

    # ---- continuous fit: the general orthogonal correction (no assumption that
    # the mapping is axis-aligned). Orthogonal Procrustes on the unit responses.
    X = np.column_stack([r / np.linalg.norm(r) for r, _, _ in obs])
    Y = np.column_stack([R.T @ d for _, R, d in obs])
    U, _, Vt = np.linalg.svd(Y @ X.T)
    M_fit = U @ Vt
    det_fit = float(np.linalg.det(M_fit))
    print("\nbest-fit sensor correction M (corrected = M @ raw):")
    for row in M_fit:
        print("   " + "  ".join(f"{v:+6.3f}" for v in row))
    print(f"  det = {det_fit:+.3f}   ({'MIRROR' if det_fit < 0 else 'rotation'})")
    errs_fit = [_ang_deg(R @ M_fit @ r, d) for r, R, d in obs]
    print("  per-push residual: " + ", ".join(
        f"{lbl}={e:.1f}deg" for lbl, e in zip(("x", "y", "z"), errs_fit)))

    # ---- discrete fit: the nearest signed permutation (a much cleaner config
    # value IF one fits; the continuous M above is the fallback).
    rows = []
    for M, perm, sg in _signed_perms():
        errs = [_ang_deg(R @ M @ r, d) for r, R, d in obs]
        rows.append((float(np.mean(errs)), errs, perm, sg, float(np.linalg.det(M))))
    rows.sort(key=lambda x: x[0])
    print(f"\n{'mean':>6} {'x':>6} {'y':>6} {'z':>6}  corrected <- raw            det")
    for mean, errs, perm, sg, det in rows[:5]:
        mapping = ", ".join(f"{'xyz'[i]}={'+-'[s < 0]}{'xyz'[c]}"
                            for i, (c, s) in enumerate(zip(perm, sg)))
        print(f"{mean:6.1f} {errs[0]:6.1f} {errs[1]:6.1f} {errs[2]:6.1f}  "
              f"{mapping:24s}  {det:+.0f}")

    cur = np.diag(np.asarray(cfg.WRENCH_AXIS_SIGN, dtype=float))
    cur_errs = [_ang_deg(R @ cur @ r, d) for r, R, d in obs]
    print(f"\ncurrent WRENCH_AXIS_SIGN {tuple(cfg.WRENCH_AXIS_SIGN)}: "
          + ", ".join(f"{lbl} err {e:.1f}deg" for lbl, e in zip("xyz", cur_errs))
          + f", det {np.linalg.det(cur):+.0f}")

    best = rows[0]
    if det_fit < 0 and np.linalg.det(cur) > 0:
        logger.warning("[probe] the correction is a MIRROR (det {:+.2f}) but the "
                       "current WRENCH_AXIS_SIGN has det {:+.0f} — a positive-det "
                       "sign array CANNOT express this mapping, whatever the "
                       "values. The product of the three signs must be negative.",
                       det_fit, np.linalg.det(cur))
    if best[0] > 20.0:
        logger.warning("[probe] no signed permutation fits (best {:.0f}deg) — the "
                       "sensor is not axis-aligned with the EE frame. Use the "
                       "continuous M above (the wrench code needs a 3x3, not a "
                       "sign array), or re-run if the pushes were sloppy.", best[0])
    elif best[2] != (0, 1, 2):
        logger.warning("[probe] the best mapping SWAPS axes ({}) — a diagonal "
                       "WRENCH_AXIS_SIGN cannot express this either.", best[2])


def _main() -> None:
    from dexcontrol.robot import Robot

    with Robot() as bot:
        arm = ArmMover(robot=bot, side=cfg.ARM_SIDE, ee_frame=cfg.EE_FRAME)
        w = getattr(arm._arm, "wrench_sensor", None)
        if w is None:
            logger.error("[probe] no wrench sensor on the {} arm", cfg.ARM_SIDE)
            return
        (_fit if "--fit" in sys.argv else _live)(arm, w)
        logger.info("[probe] done")


if __name__ == "__main__":
    _main()
