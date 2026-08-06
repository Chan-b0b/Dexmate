#!/usr/bin/env python3
"""Live FiLM counterfactual authority probe.

Moves the suction arm through clearances above a detected case (negative
clearances DO press the cup into the case). At each pose it freezes one real
RGB/depth/state observation and predicts under counterfactuals. Predicted
actions are logged only and are NEVER sent to the robot.

Two counterfactual families:
  c-hat forcing (FiLM model only) — full COHERENT anchor vectors (hover /
    preseal / sealed, all channels incl. fmag forced together so contact=1
    never pairs with a hover-level fmag) + per-channel fz +-N sweeps.
  RAW-state swaps (run on BOTH models) — fz +-N / measured first-contact /
    sealed wrench written into the state itself: the naive baseline has no
    c-hat, so the state IS its counterfactual; for the FiLM model the same
    swap produces a coherent c-hat via the real computation path. This is the
    live replication of probe_state_authority.py.

--baseline-checkpoint <naive ckpt> runs the vanilla baseline on the SAME
frozen observations for a film-vs-naive comparison (…_vs_naive.png).

Before the clearance sweep the arm holds the pre-descent hover pose and
re-anchors the FiLM offsets against the live wrench (--baseline-hover; same
mechanism as run_policy --film-auto-baseline) — FILM_* envs must therefore be
the TRAINING values, never a pre-corrected env file.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger

VLA_DIR = Path(__file__).resolve().parent
REPO_DIR = VLA_DIR.parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(VLA_DIR))

import film_contact  # noqa: E402
import run_policy as rp  # noqa: E402


# new_embed_prefix resolves this module-global function on every forward. The
# override therefore changes only c-hat while the captured observation stays fixed.
_FORCE_C: list[float] | None = None
_REAL_CONDITION_FN = film_contact._condition_from_state


def _condition_override(model, state):
    if _FORCE_C is None:
        return _REAL_CONDITION_FN(model, state)
    value = torch.tensor(_FORCE_C, dtype=torch.float32, device=state.device)
    return value.expand(state.shape[0], -1).clone()


film_contact._condition_from_state = _condition_override


@dataclass
class Args:
    checkpoint: Path | None = None
    baseline_checkpoint: Path | None = None  # vanilla (naive) ckpt, same frozen obs
    output_dir: Path = VLA_DIR / "live_film_probes"
    layers: int | None = None
    clearances: tuple[float, ...] | None = None
    fz_deltas_n: tuple[float, ...] | None = None
    # self-anchor height (m above contact): before the clearance sweep, hold the
    # PRE-DESCENT HOVER pose and re-anchor the FiLM offsets there (the 08-04 run
    # showed env-file offsets measured at another pose mis-anchor c-hat by ~0.7 N).
    # Train stats at this pose (|F| 4.62 / fz 2.04) match _film_auto_baseline's
    # ep-start anchors within 0.1 N. 0 disables.
    baseline_hover: float = 0.30
    seed: int = 0
    go: bool = False


# state layout (ObsBuilder): pos3 quat4 suction(7) seal(8) wrench fx..tz(9:15)
SEAL_IDX, WRENCH_LO, WRENCH_HI, FZ_IDX = 8, 9, 15, 11
# Measured 0729-val medians for the raw-state swaps (probe_state_authority.py
# conventions: first-contact = fz>3.0N trigger, 18 frames; sealed = 38 frames):
FC_WRENCH = (1.32, 5.63, 3.65, 0.02, -0.48, -0.46)       # |F| = 6.8 N
SEALED_WRENCH = (0.47, 5.15, 4.98, -0.09, -0.49, -0.50)  # |F| = 7.2 N
# Train pre-descent hover wrench median (0729 train, 98 eps) — reference for
# shifting the ABSOLUTE swap vectors onto today's F/T bias: without the shift,
# a drift-re-anchored FiLM model reads the swap at (training dose - drift)
# while the naive baseline gets the exact training template -> asymmetric
# comparison (the 08-06 run measured exactly this).
TRAIN_HOVER_WRENCH = (0.80, 3.99, 2.04, 0.15, -0.59, -0.36)  # |F| = 4.55 N
# Measured 0729-train c-hat anchors (RECAL calibration: F0/fmag 5.5/1, fz 3.0/0.7).
# A scenario forces the FULL vector so channels never contradict each other;
# channels without an anchor entry (e.g. dfmag) keep their real value.
# NOTE: values are calibration-specific — older 3-channel generations used
# different offsets; re-derive before probing non-recal checkpoints.
CHAT_ANCHORS = {
    "hover":   {"contact": 0.0, "fz": -1.43, "fmag": -0.9, "seal": 0.0},
    "preseal": {"contact": 0.4, "fz": 0.86,  "fmag": 0.4,  "seal": 0.0},
    "sealed":  {"contact": 1.0, "fz": 2.29,  "fmag": 1.2,  "seal": 1.0},
}


def _forced_anchor(real_c: list[float], cond_names: list[str], anchor: dict) -> list[float]:
    return [float(anchor.get(name, real_c[i])) for i, name in enumerate(cond_names)]


def _state_variants(state: np.ndarray, fz_deltas_n,
                    drift6: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """RAW-state counterfactuals, applied identically to both models.
    drift6 = live hover wrench median - TRAIN_HOVER_WRENCH: the absolute
    fc/sealed swap vectors are shifted onto today's bias so both models get
    the same PHYSICAL counterfactual (fz+-N deltas are relative already)."""
    d = np.zeros(6) if drift6 is None else np.asarray(drift6, dtype=float)
    out = {}
    for dn in fz_deltas_n:
        s = state.copy()
        s[FZ_IDX] += float(dn)
        out[f"st_fz{dn:+g}N"] = s
    s = state.copy()
    s[WRENCH_LO:WRENCH_HI] = np.asarray(FC_WRENCH) + d
    out["st_firstcontact"] = s
    s = state.copy()
    s[WRENCH_LO:WRENCH_HI] = np.asarray(SEALED_WRENCH) + d
    s[SEAL_IDX] = 1.0
    out["st_sealed"] = s
    return out


def _capture(bot, mover, seal, ob):
    torso_q = bot.torso.get_joint_pos()
    left_q = bot.left_arm.get_joint_pos()
    right_q = bot.right_arm.get_joint_pos()
    ws = getattr(mover._arm, "wrench_sensor", None)
    wrench6 = (np.asarray(ws.get_state()["wrench"], dtype=float)[:6]
               if ws is not None else np.zeros(6))
    from LGES.ik_demo.drivers import suction_io
    suction_on = suction_io.is_suction_commanded_on()
    sealed = bool(seal.is_sealed()) if seal is not None else False
    state = ob.state(torso_q, left_q, right_q, wrench6, suction_on, sealed)
    rgb = rp._grab_rgb(bot)
    depth_m = rp._grab_depth(bot)
    return state, rgb, depth_m, wrench6


def _predict(policy, pre, post, ob, state, rgb, depth_m, instruction, forced_c, seed,
             expect_film: bool = True):
    global _FORCE_C
    _FORCE_C = forced_c
    policy.reset()
    torch.manual_seed(seed)
    pred = rp.predict(policy, pre, post, state, ob.image(rgb), instruction,
                      ob.depth_image(depth_m))
    diag = rp._film_diagnostics(policy)
    if diag is None and expect_film:
        raise RuntimeError("FiLM diagnostics unavailable after prediction")
    return pred, diag


def _action_summary(pred, state, abs_action: bool, suction_idx: int):
    if abs_action:
        dpos = np.asarray(pred[:3], dtype=float) - np.asarray(state[:3], dtype=float)
    else:
        dpos = np.asarray(pred[:3], dtype=float)
    return {
        "action": [float(v) for v in pred],
        "dpos_m": [float(v) for v in dpos],
        "suction": float(pred[suction_idx]),
    }


def _plot(result: dict, out: Path):
    """Summarize each counterfactual as an action delta from real across poses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = result["poses"]
    scenarios = [name for name in poses[0]["scenarios"] if name != "real"]
    metrics = {name: {"dz": [], "suction": [], "translation": [], "rotation": []}
               for name in scenarios}
    abs_action = result.get("action_space") == "absolute"

    def rotation_delta(action, real_action):
        if not abs_action:
            return float(np.linalg.norm(np.asarray(action[3:6]) - np.asarray(real_action[3:6])))
        q0 = np.asarray(real_action[3:7], dtype=float)
        q1 = np.asarray(action[3:7], dtype=float)
        q0 /= np.linalg.norm(q0)
        q1 /= np.linalg.norm(q1)
        return float(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0)))

    for pose in poses:
        real = pose["scenarios"]["real"]
        real_dpos = np.asarray(real["dpos_m"], dtype=float)
        for name in scenarios:
            row = pose["scenarios"][name]
            delta_pos = np.asarray(row["dpos_m"], dtype=float) - real_dpos
            metrics[name]["dz"].append(float(delta_pos[2] * 1000))
            metrics[name]["suction"].append(float(row["suction"] - real["suction"]))
            metrics[name]["translation"].append(float(np.linalg.norm(delta_pos) * 1000))
            metrics[name]["rotation"].append(
                rotation_delta(row["action"], real["action"]) * 1000)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), constrained_layout=True)
    specs = (("dz", "delta dz vs real (mm)"),
             ("suction", "delta suction vs real"),
             ("translation", "translation action distance vs real (mm)"),
             ("rotation", "rotation action distance vs real (mrad)"))
    colors = plt.get_cmap("tab10")(np.arange(len(scenarios)) % 10)
    for ax, (key, ylabel) in zip(axes, specs):
        values = [metrics[name][key] for name in scenarios]
        boxes = ax.boxplot(values, labels=scenarios, showmeans=True, patch_artist=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for i, vals in enumerate(values, start=1):
            jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(i + jitter, vals, color=colors[i - 1], s=28, zorder=3)
            ax.text(i, max(vals) if vals else 0.0, f"mean {np.mean(vals):+.3g}",
                    ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Counterfactual action change vs real ({len(poses)} frozen poses)")
    fig.savefig(out, dpi=160)
    plt.close(fig)

def _plot_pose(pose: dict, out: Path, action_space: str = "delta"):
    """Plot one clearance separately so overlapping scenarios stay readable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(pose["scenarios"])
    rows = [pose["scenarios"][name] for name in scenarios]
    x = np.arange(len(scenarios))
    colors = plt.get_cmap("tab10")(x % 10)
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), constrained_layout=True)

    axes[0].bar(x, [row["dpos_m"][2] * 1000 for row in rows], color=colors)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("predicted dz (mm)")

    axes[1].bar(x, [row["suction"] for row in rows], color=colors)
    axes[1].axhline(0.5, color="black", linewidth=0.7)
    axes[1].set_ylabel("predicted suction")

    real = pose["scenarios"]["real"]
    real_dpos = np.asarray(real["dpos_m"], dtype=float)
    translation_delta = [np.linalg.norm(np.asarray(row["dpos_m"]) - real_dpos) * 1000
                         for row in rows]
    if action_space == "absolute":
        real_q = np.asarray(real["action"][3:7], dtype=float)
        real_q /= np.linalg.norm(real_q)
        rotation_delta = []
        for row in rows:
            q = np.asarray(row["action"][3:7], dtype=float)
            q /= np.linalg.norm(q)
            rotation_delta.append(2 * np.arccos(np.clip(abs(np.dot(real_q, q)), 0, 1)) * 1000)
    else:
        real_rot = np.asarray(real["action"][3:6], dtype=float)
        rotation_delta = [np.linalg.norm(np.asarray(row["action"][3:6]) - real_rot) * 1000
                          for row in rows]
    width = 0.38
    axes[2].bar(x - width / 2, translation_delta, width,
                label="translation distance", color="tab:blue")
    rotation_ax = axes[2].twinx()
    rotation_ax.bar(x + width / 2, rotation_delta, width,
                    label="rotation distance", color="tab:orange")
    axes[2].set_ylabel("translation delta vs real (mm)")
    rotation_ax.set_ylabel("rotation delta vs real (mrad)")
    axes[2].legend(loc="upper left")
    rotation_ax.legend(loc="upper right")

    cond_names = rows[0]["film"]["cond_names"]
    c_hat = np.asarray([row["film"]["c_hat"] for row in rows], dtype=float)
    image = axes[3].imshow(c_hat, aspect="auto", cmap="coolwarm")
    axes[3].set_xticks(np.arange(len(cond_names)), cond_names)
    axes[3].set_yticks(x, scenarios)
    axes[3].set_xlabel("FiLM condition channel")
    axes[3].set_ylabel("counterfactual scenario")
    for row_i in range(c_hat.shape[0]):
        for col_i in range(c_hat.shape[1]):
            axes[3].text(col_i, row_i, f"{c_hat[row_i, col_i]:.2f}",
                         ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=axes[3], label="c_hat")

    for ax in axes[:3]:
        ax.set_xticks(x, scenarios, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    clearance_cm = pose["clearance_m"] * 100
    fig.suptitle(f"Live FiLM counterfactuals at {clearance_cm:g} cm clearance")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_compare(result: dict, out: Path):
    """film vs naive on the SAME frozen observations — dz ONLY (2026-08-04):
    real-scenario dz across clearances + per-scenario dz shift vs real."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = [p for p in result["poses"] if p.get("baseline_scenarios")]
    if not poses:
        return
    cl = [p["clearance_m"] * 100 for p in poses]
    shared = [n for n in poses[0]["baseline_scenarios"] if n != "real"]
    models = (("film", "scenarios", "tab:blue"), ("naive", "baseline_scenarios", "tab:orange"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    for label, key, color in models:
        axes[0].plot(cl, [p[key]["real"]["dpos_m"][2] * 1000 for p in poses],
                     marker="o", label=label, color=color)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("clearance (cm)  [negative = pressing]")
    axes[0].set_ylabel("predicted dz, real obs (mm)")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    x = np.arange(len(shared))
    width = 0.38
    for off, (label, key, color) in zip((-width / 2, width / 2), models):
        ddz = [np.mean([(p[key][n]["dpos_m"][2] - p[key]["real"]["dpos_m"][2]) * 1000
                        for p in poses]) for n in shared]
        axes[1].bar(x + off, ddz, width, label=label, color=color)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, shared, rotation=25, ha="right")
    axes[1].set_ylabel("mean delta dz vs real (mm)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle(f"film vs naive, dz on identical frozen observations "
                 f"({len(poses)} poses, raw-state counterfactuals)")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _preflight(mover, targets, rpy, reach_tol):
    seed = np.asarray(mover._arm.get_joint_pos(), dtype=float)
    rows = []
    for label, pos in targets:
        sol = mover.solve_pose(pos, rpy, seed=seed, min_motion=True)
        ok = sol.pos_err_m <= reach_tol and sol.in_limits and not sol.in_collision
        rows.append((label, pos, sol, ok))
        if ok:
            seed = sol.q
    return rows


def main():
    args = tyro.cli(Args)
    if not args.go:
        raise SystemExit("This probe moves the real arm. Re-run with --go to continue.")

    from dexcontrol.core.config import get_robot_config
    from dexcontrol.robot import Robot
    from LGES.ik_demo import config as ikcfg
    from LGES.ik_demo.chassis_sequence import (detect, _center_from_det, _view_park,
                                               set_head_pitch)
    from LGES.ik_demo.drivers import suction_io
    from LGES.ik_demo.go_home import both_arms_home
    from LGES.ik_demo.suction import SuctionMover

    checkpoint = args.checkpoint or rp.latest_checkpoint()
    clearances = tuple(args.clearances or ikcfg.VLA_FILM_PROBE_CLEARANCES_M)
    # if not clearances or min(clearances) < 0.03:
    #     raise SystemExit("clearances must be >= 0.03 m for this non-contact probe")
    clearances = tuple(sorted((float(v) for v in clearances), reverse=True))
    layers = int(ikcfg.SRC_LAYERS_REMAINING if args.layers is None else args.layers)
    fz_deltas_n = tuple(args.fz_deltas_n or ikcfg.VLA_FILM_PROBE_FZ_DELTAS_N)

    logger.warning("MOVES THE REAL LEFT ARM through {} non-contact probe poses.", len(clearances))
    logger.warning("Policy actions are prediction-only and are NEVER commanded.")
    logger.warning("Keep a hand on the E-Stop. Lowest clearance is {:.0f} mm.",
                   min(clearances) * 1000)
    if input("Continue? [y/N]: ").strip().lower() != "y":
        print("cancelled")
        return

    policy, pre, post = rp.load_policy(checkpoint, film=True)
    policy.config.n_action_steps = 1
    ob = rp.ObsBuilder()
    ob.df_channel = int(policy.config.robot_state_feature.shape[0]) == 16
    abs_action = int(policy.config.action_feature.shape[0]) == 8
    suction_idx = 7 if abs_action else 6
    instruction = rp.TASKS["case_pick"]["instruction"]

    base = None
    if args.baseline_checkpoint:
        # Loaded AFTER the film policy, i.e. under the patched VLAFlowMatching —
        # _film_cond=None opts this instance out of c-hat + force-masking so it
        # behaves exactly like a vanilla checkpoint (guard in film_contact).
        bpolicy, bpre, bpost = rp.load_policy(args.baseline_checkpoint, film=False)
        bpolicy.config.n_action_steps = 1
        if hasattr(bpolicy.model, "_film_cond"):
            bpolicy.model._film_cond = None
        b_abs = int(bpolicy.config.action_feature.shape[0]) == 8
        base = (bpolicy, bpre, bpost, b_abs, 7 if b_abs else 6)
        logger.info("baseline (vanilla) loaded: {}", args.baseline_checkpoint)

    robot_configs = get_robot_config()
    robot_configs.enable_sensor("head_camera")
    robot_configs.sensors["head_camera"].transport = "zenoh"
    seal = None
    moved = False
    result = None
    with Robot(configs=robot_configs) as bot, SuctionMover(bot) as mover:
        try:
            if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
                raise RuntimeError("head camera did not become active")
            release = mover.software_estop_active()
            if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
                return
            if not mover.ensure_ready(release_estop=release):
                raise RuntimeError("arm not ready")

            suction_io.suction_off()
            set_head_pitch(bot, angle=30.0)
            both_arms_home(bot, left=mover)
            moved = True
            _view_park(mover, "live-film-probe")
            det = detect(bot, layers)
            if det is None or not det.found:
                raise RuntimeError("case detection failed")
            center = _center_from_det(det)
            pick_pose = ikcfg.resolve_poses(center)["CASE_PICK"]
            x, y = float(pick_pose[0]), float(pick_pose[1])
            rpy = tuple(float(v) for v in pick_pose[3:6])
            contact_ee_z = float(det.top_face_z) + float(ikcfg.SUCTION_LENGTH_M)
            targets = [("transport", (x, y, float(ikcfg.SAFE_TRANSPORT_Z)))]
            if args.baseline_hover:
                targets += [("baseline_hover",
                             (x, y, contact_ee_z + float(args.baseline_hover)))]
            targets += [(f"clearance_{c:.3f}", (x, y, contact_ee_z + c))
                        for c in clearances]
            checks = _preflight(mover, targets, rpy, float(ikcfg.REACH_TOL_M))
            for label, pos, sol, ok in checks:
                print(f"  IK {label:>18}: {'OK' if ok else 'FAIL'} target={np.round(pos, 4)} "
                      f"err={sol.pos_err_m*1000:.1f}mm limits={sol.in_limits} "
                      f"collision={sol.in_collision}")
            if not all(row[3] for row in checks):
                raise RuntimeError("IK preflight failed; no probe motion started")

            from LGES.ik_demo.drivers.suction_io import VacuumMonitor
            try:
                seal = VacuumMonitor()
                seal.start()
            except Exception as e:  # noqa: BLE001
                logger.warning("vacuum monitor unavailable: {}", e)
                seal = None

            if mover.move_ee(targets[0][1], rpy, quiet=True) is None:
                raise RuntimeError("transport approach failed")

            film_baseline = None
            if args.baseline_hover:
                # re-anchor F0/FMAG_OFF/FZ_OFF at the pre-descent hover, right next
                # to the region being probed — FILM_* envs must hold TRAIN values
                # (no pre-corrected env file; _film_auto_baseline stashes them).
                hover = (x, y, contact_ee_z + float(args.baseline_hover))
                if mover.move_ee(hover, rpy, quiet=True) is None:
                    logger.warning("baseline hover move failed — offsets stay at env values")
                else:
                    time.sleep(float(ikcfg.VLA_FILM_PROBE_SETTLE_S))
                    entry_sealed = bool(seal.is_sealed()) if seal is not None else False
                    film_baseline = rp._film_auto_baseline(mover, policy, entry_sealed)

            swap_drift = None
            if film_baseline and film_baseline.get("wrench_med"):
                swap_drift = (np.asarray(film_baseline["wrench_med"], dtype=float)
                              - np.asarray(TRAIN_HOVER_WRENCH, dtype=float))
                print(f"  swap drift (live hover - train hover): {np.round(swap_drift, 2)}")

            result = {
                "film_baseline": film_baseline,
                "swap_drift": ([round(float(v), 3) for v in swap_drift]
                               if swap_drift is not None else None),
                "checkpoint": str(checkpoint),
                "baseline_checkpoint": (str(args.baseline_checkpoint)
                                        if args.baseline_checkpoint else None),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "detected_center_xyzyaw": [float(v) for v in center],
                "top_face_z": float(det.top_face_z),
                "contact_ee_z": contact_ee_z,
                "action_space": "absolute" if abs_action else "delta",
                "seed": args.seed,
                "fz_deltas_n": [float(v) for v in fz_deltas_n],
                "fz_tau": (float(policy.model._fz_tau)
                           if hasattr(policy.model, "_fz_tau") else None),
                "chat_anchors": CHAT_ANCHORS,
                "fc_wrench": list(FC_WRENCH),
                "sealed_wrench": list(SEALED_WRENCH),
                "poses": [],
            }

            for clearance in clearances:
                answer = input(f"\nENTER: move to {clearance*100:.1f} cm clearance; q: finish > ")
                if answer.strip().lower() == "q":
                    break
                target = (x, y, contact_ee_z + clearance)
                if mover.move_ee(target, rpy, quiet=False) is None:
                    raise RuntimeError(f"move failed at clearance {clearance:.3f} m")
                time.sleep(float(ikcfg.VLA_FILM_PROBE_SETTLE_S))
                state, rgb, depth_m, wrench6 = _capture(bot, mover, seal, ob)

                real_pred, real_diag = _predict(policy, pre, post, ob, state, rgb, depth_m,
                                                instruction, None, args.seed)
                cond_names = real_diag["cond_names"]
                real_c = real_diag["c_hat"]
                # c-hat forcing (film only): full coherent anchor vectors
                patterns = {"real": None}
                for name, anchor in CHAT_ANCHORS.items():
                    patterns[name] = _forced_anchor(real_c, cond_names, anchor)
                if "fz" in cond_names:
                    fz_i = cond_names.index("fz")
                    fz_tau = float(policy.model._fz_tau)
                    for delta_n in fz_deltas_n:
                        forced = list(real_c)
                        forced[fz_i] = real_c[fz_i] + float(delta_n) / fz_tau
                        patterns[f"fz_{delta_n:+g}N"] = forced
                svars = _state_variants(state, fz_deltas_n, drift6=swap_drift)
                scenarios = {}
                for name, forced_c in patterns.items():
                    if name == "real":
                        pred, diag = real_pred, real_diag
                    else:
                        pred, diag = _predict(policy, pre, post, ob, state, rgb, depth_m,
                                              instruction, forced_c, args.seed)
                    row = _action_summary(pred, state, abs_action, suction_idx)
                    row["film"] = diag
                    scenarios[name] = row
                    print(f"  {name:>16}: c={np.round(diag['c_hat'], 3)} "
                          f"dz={row['dpos_m'][2]*1000:+.2f}mm suction={row['suction']:.3f} "
                          f"gamma_rms={diag['gamma']['rms']:.4f} beta_rms={diag['beta']['rms']:.4f}")
                # raw-state swaps (film): coherent c-hat via the real path
                for name, s2 in svars.items():
                    pred, diag = _predict(policy, pre, post, ob, s2, rgb, depth_m,
                                          instruction, None, args.seed)
                    row = _action_summary(pred, s2, abs_action, suction_idx)
                    row["film"] = diag
                    scenarios[name] = row
                    print(f"  {name:>16}: c={np.round(diag['c_hat'], 3)} "
                          f"dz={row['dpos_m'][2]*1000:+.2f}mm suction={row['suction']:.3f}")

                baseline_scenarios = None
                if base is not None:
                    bpolicy, bpre, bpost, b_abs, b_sidx = base
                    baseline_scenarios = {}
                    for name, s2 in {"real": state, **svars}.items():
                        pred, diag = _predict(bpolicy, bpre, bpost, ob, s2, rgb, depth_m,
                                              instruction, None, args.seed,
                                              expect_film=False)
                        row = _action_summary(pred, s2, b_abs, b_sidx)
                        row["film"] = diag  # None: vanilla baseline has no c-hat
                        baseline_scenarios[name] = row
                        print(f"  [naive] {name:>16}: dz={row['dpos_m'][2]*1000:+.2f}mm "
                              f"suction={row['suction']:.3f}")

                result["poses"].append({
                    "clearance_m": clearance,
                    "target_ee_pos": [float(v) for v in target],
                    "observed_ee_pos": [float(v) for v in state[:3]],
                    "wrench": [float(v) for v in wrench6],
                    "real_c_hat": real_c,
                    "scenarios": scenarios,
                    "baseline_scenarios": baseline_scenarios,
                })
        finally:
            global _FORCE_C
            _FORCE_C = None
            if seal is not None:
                seal.stop()
            if moved:
                try:
                    pos, rpy_now = mover.current_ee_pose()
                    if float(pos[2]) < float(ikcfg.SAFE_TRANSPORT_Z) - 1e-3:
                        mover.move_ee_vertical(float(ikcfg.SAFE_TRANSPORT_Z), tuple(rpy_now))
                    _view_park(mover, "live-film-probe-finish")
                except Exception as e:  # noqa: BLE001
                    logger.error("safe retreat failed: {}", e)

    if result is None or not result["poses"]:
        print("no probe poses captured")
        return
    save_dir = args.output_dir / Path(checkpoint).name
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S") + "_live_film_authority"
    json_path = save_dir / f"{stem}.json"
    png_path = save_dir / f"{stem}.png"
    json_path.write_text(json.dumps(result, indent=2))
    _plot(result, png_path)
    pose_pngs = []
    for pose in result["poses"]:
        clearance_label = f"{pose['clearance_m'] * 100:g}".replace(".", "p")
        pose_path = save_dir / f"{stem}_{clearance_label}cm.png"
        _plot_pose(pose, pose_path, action_space=result.get("action_space", "delta"))
        pose_pngs.append(pose_path)
    print(f"\nresult: {json_path}\nsummary: {png_path}")
    if result.get("baseline_checkpoint"):
        cmp_path = save_dir / f"{stem}_vs_naive.png"
        _plot_compare(result, cmp_path)
        print(f"compare: {cmp_path}")
    for pose_path in pose_pngs:
        print(f"pose:    {pose_path}")


if __name__ == "__main__":
    main()
"""
# 0729 recal round (film vs naive on the same frozen observations):
FILM_COND=contact,fmag,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
FILM_F0=5.5 FILM_TAU=1 FILM_FMAG_OFF=5.5 FILM_FMAG_TAU=1 \
FILM_FZ_OFF=3.0 FILM_FZ_TAU=0.7 \
FILM_DATASET=lges_case_pick_0729 \
python probe_film_authority_live.py --go \
  --clearances 0.05 0.04 0.03 0.02 0.01 0.00 -0.01 -0.02 -0.03 -0.04 \
  --checkpoint Chanho-Lee/smolvla_film_0729_prefix_mask1_recal_fromnaive \
  --baseline-checkpoint Chanho-Lee/smolvla_naive_0729 \
  --fz-deltas-n -6 -3 3 6
"""
