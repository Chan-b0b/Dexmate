#!/usr/bin/env python3
"""Passive reward probe: runs chassis_sequence.py's REAL pick&place routine while a
background thread samples observation.state at FPS and scores each consecutive
(state, retro-derived action, next_state) transition with the frozen case_pick ICM.

chassis_sequence.py is not an RL rollout -- pick()/place()/strafe() are blocking
calls, not a per-tick delta-pose policy, so there's no logged "action" to reuse.
Instead this reconstructs the executed action the same way run_policy.py's
self_test() does for its "delta-integration convention" check: dpos = next_pos -
pos, drot = rotvec(q_next * q_cur^-1), suction = the state's own suction bit.

Reports TWO signals per tick, since reward()'s exp(-eta*error) is calibrated to
case_pick's OWN held-out demo error scale (p95 -> r=0.8) and can saturate near 0
for a structurally different task/pose region, hiding relative differences:
  - reward:        the real, calibrated reversed-curiosity reward (what actor.py uses)
  - resid_norm:    ||next_feature - predicted_feature||, uncalibrated, same units as
                   `state` -- doesn't saturate, so it still shows which ticks were
                   relatively more/less demo-like even when reward is uniformly low

EACH is also reported "_corr" (wrench-offset corrected): a live-rollout-vs-training
z-score comparison (2026-07-16) found position/orientation match the case_pick_0708
demo distribution within 1 sigma, but 5 of 6 wrench dims sit 2.5-4.8 sigma off --
consistent with the F/T sensor's raw baseline having drifted since those demos were
recorded. WRENCH_OFFSET below is measured_live_mean - training_mean on that data; the
_corr columns subtract it before scoring, as a stopgap to test whether that alone
explains the low reward. It is NOT a permanent fix (re-zeroing the sensor or
recalibrating the ICM on fresh demos would be) -- if _corr comes back near the
held-out demo range (~0.6-1.0) while raw stays low, that confirms the hypothesis.

Prints reward/resid_norm only -- no trained residual-policy critic checkpoint exists
yet (rl_ckpt is empty), so an advantage number here would just reflect a freshly
initialized value function, not anything learned.

Does NOT write into Research/residual_rl/rl_buffer: this is a passive probe on an
unrelated task (multi-item chassis pick&place) and would corrupt the case_pick
replay buffer (single left-arm position, delta-pose task) if logged there. Per-tick
records instead go to --out (default: a timestamped file under ./chassis_probe_logs/,
gitignored) as JSONL: {"t", "state", "action", "reward", "resid_norm"}.

Run with the vla_venv python:
  /home/dexmate/vla_venv/bin/python Research/residual_rl/chassis_reward_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))
sys.path.insert(0, str(HERE))

from run_policy import ObsBuilder, FPS  # noqa: E402
from icm_reward import ICMRewarder, WRENCH_OFFSET_0716 as WRENCH_OFFSET  # noqa: E402


def _action_from_states(prev_state: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Retroactive 7-d delta-pose action from two consecutive samples -- same
    convention as run_policy.py's integrate()/self_test() (dpos = next - cur,
    drot = rotvec(q_next * q_cur^-1)); suction carried from the later state."""
    dpos = state[0:3] - prev_state[0:3]
    w0, x0, y0, z0 = prev_state[3:7]
    w1, x1, y1, z1 = state[3:7]
    q_cur = Rotation.from_quat([x0, y0, z0, w0])
    q_nxt = Rotation.from_quat([x1, y1, z1, w1])
    drot = (q_nxt * q_cur.inv()).as_rotvec()
    return np.concatenate([dpos, drot, [state[7]]]).astype(np.float32)


def run_probe(icm_checkpoint: str, icm_filename: str, out_path: Path):
    from dexcontrol.robot import Robot
    from dexcontrol.core.config import get_robot_config
    from LGES.ik_demo.suction import SuctionMover
    from LGES.ik_demo.drivers import suction_io
    from LGES.ik_demo.go_home import both_arms_home
    from LGES.ik_demo import chassis_sequence as cs

    ob = ObsBuilder()
    rewarder = ICMRewarder(icm_checkpoint, filename=icm_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w")
    print(f"logging per-tick records -> {out_path}")

    robot_configs = get_robot_config()
    robot_configs.enable_sensor("head_camera")
    robot_configs.sensors["head_camera"].transport = "zenoh"
    with Robot(configs=robot_configs) as bot, SuctionMover(bot) as mover:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("  (head camera may not be active)")
        cs.set_head_pitch(bot, angle=30.0)

        release = mover.software_estop_active()
        if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
            return
        if not mover.ensure_ready(release_estop=release):
            print("arm not ready — aborting")
            return

        print("\n  *** THIS RUNS chassis_sequence's REAL pick&place "
              "(chassis + left arm + suction). Keep a hand on the e-stop. ***")
        try:
            input("  >>> ENTER to authorize motion (Ctrl-C to cancel) <<< ")
        except (KeyboardInterrupt, EOFError):
            print("cancelled — no motion.")
            return

        seal = None
        try:
            seal = suction_io.VacuumMonitor()
            seal.start()
        except Exception as e:  # noqa: BLE001
            print(f"  (vacuum monitor unavailable: {e}; seal -> 0)")

        rewards: list[float] = []
        rewards_corr: list[float] = []
        resid_norms: list[float] = []
        resid_norms_corr: list[float] = []
        stop_flag = threading.Event()

        def sample_loop():
            dt = 1.0 / FPS
            prev_state = None
            while not stop_flag.is_set():
                t0 = time.time()
                torso_q = bot.torso.get_joint_pos()
                left_q = np.asarray(bot.left_arm.get_joint_pos(), dtype=float)
                right_q = bot.right_arm.get_joint_pos()
                ws = getattr(mover._arm, "wrench_sensor", None)
                wrench6 = (np.asarray(ws.get_state()["wrench"], float)[:6]
                           if ws is not None else np.zeros(6))
                suction = suction_io.is_suction_commanded_on()
                sealed = bool(seal.is_sealed()) if seal else False
                state = ob.state(torso_q, left_q, right_q, wrench6, suction, sealed)
                if prev_state is not None:
                    act = _action_from_states(prev_state, state)
                    r = rewarder.reward(prev_state, act, state)
                    r_corr = rewarder.reward(prev_state, act, state, wrench_offset=WRENCH_OFFSET)
                    resid_norm, _ = rewarder.feature_residual(prev_state, act, state)
                    resid_norm_corr, _ = rewarder.feature_residual(
                        prev_state, act, state, wrench_offset=WRENCH_OFFSET)
                    rewards.append(r)
                    rewards_corr.append(r_corr)
                    resid_norms.append(resid_norm)
                    resid_norms_corr.append(resid_norm_corr)
                    out_f.write(json.dumps({
                        "t": len(rewards) - 1,
                        "state": [float(x) for x in prev_state],
                        "action": [float(x) for x in act],
                        "reward": r,
                        "reward_corr": r_corr,
                        "resid_norm": resid_norm,
                        "resid_norm_corr": resid_norm_corr,
                    }) + "\n")
                    out_f.flush()
                    print(f"\r[{len(rewards):5d}] reward={r:.3f}->{r_corr:.3f} "
                          f"resid_norm={resid_norm:.3f}->{resid_norm_corr:.3f} "
                          f"z={state[2]:.3f} suc={'Y' if suction else '.'} "
                          f"seal={'Y' if sealed else '.'}"
                          "\033[K", end="", flush=True)
                prev_state = state
                time.sleep(max(0.0, dt - (time.time() - t0)))

        sampler = threading.Thread(target=sample_loop, daemon=True)
        sampler.start()
        try:
            print("-> both arms safe home")
            both_arms_home(bot, left=mover)
            ok = cs.run(bot, mover)
            print(f"\nchassis_sequence {'OK' if ok else 'FAILED'}")
        finally:
            stop_flag.set()
            sampler.join(timeout=2.0)
            if seal is not None:
                seal.stop()
            out_f.close()

        if rewards:
            r, rc = np.asarray(rewards), np.asarray(rewards_corr)
            rn, rnc = np.asarray(resid_norms), np.asarray(resid_norms_corr)
            print(f"\n{len(r)} ticks sampled:")
            print(f"  reward:          mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
            print(f"  reward_corr:     mean={rc.mean():.3f} min={rc.min():.3f} max={rc.max():.3f}")
            print(f"  resid_norm:      mean={rn.mean():.3f} min={rn.min():.3f} max={rn.max():.3f}")
            print(f"  resid_norm_corr: mean={rnc.mean():.3f} min={rnc.min():.3f} max={rnc.max():.3f}")
            print(f"  saved -> {out_path}")
        else:
            print("\nno ticks sampled")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icm-checkpoint", default="Chanho-Lee/icm_case_pick_0708")
    ap.add_argument("--icm-filename", default="icm_0708_proprio.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSONL output path (default: chassis_probe_logs/<timestamp>.jsonl "
                         "under this directory, gitignored -- NOT the rl_buffer)")
    args = ap.parse_args()
    out_path = args.out or (HERE / "chassis_probe_logs" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    run_probe(args.icm_checkpoint, args.icm_filename, out_path)


if __name__ == "__main__":
    main()
