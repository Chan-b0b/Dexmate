#!/usr/bin/env python3
"""Live FiLM counterfactual authority probe.

Moves the suction arm through safe, non-contact clearances above a detected case.
At each pose it freezes one real RGB/depth/state observation and runs the policy
with real, no-contact, contact, and sealed FiLM conditions. Predicted actions are
logged only and are NEVER sent to the robot.
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
    output_dir: Path = VLA_DIR / "live_film_probes"
    layers: int | None = None
    clearances: tuple[float, ...] | None = None
    fz_deltas_n: tuple[float, ...] | None = None
    seed: int = 0
    go: bool = False


def _forced_pattern(real_c: list[float], cond_names: list[str], *,
                    contact: float | None = None,
                    seal: float | None = None,
                    fz: float | None = None) -> list[float]:
    out = list(real_c)
    if contact is not None and "contact" in cond_names:
        out[cond_names.index("contact")] = contact
    if fz is not None and "fz" in cond_names:
        out[cond_names.index("fz")] = fz
    if seal is not None and "seal" in cond_names:
        out[cond_names.index("seal")] = seal
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


def _predict(policy, pre, post, ob, state, rgb, depth_m, instruction, forced_c, seed):
    global _FORCE_C
    _FORCE_C = forced_c
    policy.reset()
    torch.manual_seed(seed)
    pred = rp.predict(policy, pre, post, state, ob.image(rgb), instruction,
                      ob.depth_image(depth_m))
    diag = rp._film_diagnostics(policy)
    if diag is None:
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = result["poses"]
    scenarios = list(poses[0]["scenarios"])
    clearance_cm = np.array([p["clearance_m"] for p in poses]) * 100
    colors = {"real": "black", "no_contact": "tab:blue",
              "contact": "tab:orange", "sealed": "tab:green"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for scenario_i, scenario in enumerate(scenarios):
        rows = [p["scenarios"][scenario] for p in poses]
        color = colors.get(scenario, cmap(scenario_i % 10))
        axes[0].plot(clearance_cm, [r["dpos_m"][2] * 1000 for r in rows],
                     marker="o", label=scenario, color=color)
        axes[1].plot(clearance_cm, [r["suction"] for r in rows],
                     marker="o", label=scenario, color=color)
        axes[2].plot(clearance_cm, [r["film"]["gamma"]["rms"] for r in rows],
                     marker="o", label=f"gamma {scenario}", color=color)
        axes[2].plot(clearance_cm, [r["film"]["beta"]["rms"] for r in rows],
                     marker="x", linestyle="--", label=f"beta {scenario}", color=color)
    axes[0].axhline(0, color="0.4", linewidth=0.7)
    axes[0].set_ylabel("predicted dz (mm)")
    axes[1].set_ylabel("predicted suction")
    axes[1].axhline(0.5, color="0.4", linewidth=0.7)
    axes[2].set_ylabel("FiLM RMS")
    axes[2].set_xlabel("clearance above case top (cm)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(ncols=2)
    axes[-1].invert_xaxis()
    fig.suptitle("Live FiLM counterfactual authority")
    fig.savefig(out, dpi=160)
    plt.close(fig)



def _plot_pose(pose: dict, out: Path):
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

    width = 0.38
    axes[2].bar(x - width / 2, [row["film"]["gamma"]["rms"] for row in rows],
                width, label="gamma RMS", color="tab:blue")
    axes[2].bar(x + width / 2, [row["film"]["beta"]["rms"] for row in rows],
                width, label="beta RMS", color="tab:orange")
    axes[2].set_ylabel("FiLM RMS")
    axes[2].legend()

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
    if not clearances or min(clearances) < 0.03:
        raise SystemExit("clearances must be >= 0.03 m for this non-contact probe")
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

            result = {
                "checkpoint": str(checkpoint),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "detected_center_xyzyaw": [float(v) for v in center],
                "top_face_z": float(det.top_face_z),
                "contact_ee_z": contact_ee_z,
                "action_space": "absolute" if abs_action else "delta",
                "seed": args.seed,
                "fz_deltas_n": [float(v) for v in fz_deltas_n],
                "fz_tau": (float(policy.model._fz_tau)
                           if hasattr(policy.model, "_fz_tau") else None),
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
                patterns = {
                    "real": None,
                    "no_contact": _forced_pattern(real_c, cond_names, contact=0.0, seal=0.0),
                    "contact": _forced_pattern(real_c, cond_names, contact=1.0, seal=0.0),
                    "sealed": _forced_pattern(real_c, cond_names, contact=1.0, seal=1.0),
                }
                if "fz" in cond_names:
                    fz_i = cond_names.index("fz")
                    fz_tau = float(policy.model._fz_tau)
                    for delta_n in fz_deltas_n:
                        label = f"fz_{delta_n:+g}N"
                        patterns[label] = _forced_pattern(
                            real_c, cond_names, fz=real_c[fz_i] + float(delta_n) / fz_tau)
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
                    print(f"  {name:>10}: c={np.round(diag['c_hat'], 3)} "
                          f"dz={row['dpos_m'][2]*1000:+.2f}mm suction={row['suction']:.3f} "
                          f"gamma_rms={diag['gamma']['rms']:.4f} beta_rms={diag['beta']['rms']:.4f}")

                result["poses"].append({
                    "clearance_m": clearance,
                    "target_ee_pos": [float(v) for v in target],
                    "observed_ee_pos": [float(v) for v in state[:3]],
                    "wrench": [float(v) for v in wrench6],
                    "real_c_hat": real_c,
                    "scenarios": scenarios,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S") + "_live_film_authority"
    json_path = args.output_dir / f"{stem}.json"
    png_path = args.output_dir / f"{stem}.png"
    json_path.write_text(json.dumps(result, indent=2))
    _plot(result, png_path)
    pose_pngs = []
    for pose in result["poses"]:
        clearance_label = f"{pose['clearance_m'] * 100:g}".replace(".", "p")
        pose_path = args.output_dir / f"{stem}_{clearance_label}cm.png"
        _plot_pose(pose, pose_path)
        pose_pngs.append(pose_path)
    print(f"\nresult: {json_path}\nsummary: {png_path}")
    for pose_path in pose_pngs:
        print(f"pose:    {pose_path}")


if __name__ == "__main__":
    main()
"""
FILM_COND=contact,fz,seal FILM_INJECT=prefix FILM_MASK_FORCE=1 \
FILM_F0=6 FILM_TAU=4 FILM_FZ_TAU=5 \
FILM_DATASET=lges_case_pick_0721 \
python probe_film_authority_live.py --go \
  --clearances 0.25 0.15 0.10 0.05 0.03 \
  --checkpoint Chanho-Lee/smolvla_film_0721_prefix_mask1 \
  --fz-deltas-n -6 -3 3 6
"""
