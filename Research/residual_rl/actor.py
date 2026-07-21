#!/usr/bin/env python3
"""Online actor for residual AWR fine-tuning on top of a frozen SmolVLA checkpoint.

Reuses run_policy.py's stable model-side API (load_policy, ObsBuilder, predict,
clamp_action, integrate, IK/Mover/safety plumbing) unchanged -- this file does not
modify run_policy.py. Adds: a small Gaussian residual on the 6-d pose delta
(residual_policy.ResidualPolicy), an online reversed-curiosity reward per transition
(icm_reward.ICMRewarder), and an episode-shard replay buffer (replay_buffer.EpisodeWriter)
that learner.py trains against in a separate process.

Scope: case_pick, delta action space only (abs-action checkpoints are rejected -- the
residual is defined as a pose-delta correction, see residual_policy.py's docstring).

Operator intervention: press ENTER during --go to end the current episode right there
(same stdin-poll abort run_policy.py already uses). This is a TRUNCATION, not a penalty
-- it carries no reward of its own (the ICM already scores every transition
automatically); it only stops paying real-robot time on a rollout that's clearly not
recovering. The episode retreats to hover (not a full home) -- home happens at the start
of the next run, like run_policy.py's existing abort path.

Run with the vla_venv python (--checkpoint, --icm-checkpoint, --buffer-dir and
--residual-ckpt-dir all default to sane values -- HF repo ids for the checkpoints,
rl_buffer/rl_ckpt under this directory (gitignored) for the data):
  /home/dexmate/vla_venv/bin/python Research/residual_rl/actor.py --dry-run
  /home/dexmate/vla_venv/bin/python Research/residual_rl/actor.py --go --force-limit 8

Runs episodes back-to-back with no per-episode prompt (this is an unattended online data
collector, not an interactive demo) -- Ctrl-C is the only way to stop between episodes.
"""

#python Research/residual_rl/actor.py --checkpoint Chanho-Lee/smolvla_naive_0708 --go --force-limit 20 

import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))   # run_policy.py itself adds REPO_DIR/VLA_DIR on import
sys.path.insert(0, str(HERE))  # residual_policy / icm_reward / replay_buffer

import run_policy as rp  # noqa: E402
from run_policy import (  # noqa: E402
    ObsBuilder, TASKS, FPS, HOVER_Z, DESCEND_Z, MAX_JOINT_STEP_RAD,
    clamp_action, integrate, workspace_ok, load_policy, predict,
    _rpy_to_quat_wxyz, _box_from_detection, _baseline_force, _retreat_to_hover,
    _grab_rgb, _grab_depth, latest_checkpoint,
)
from icm_reward import ICMRewarder, WRENCH_OFFSET_0716  # noqa: E402
from replay_buffer import EpisodeWriter  # noqa: E402
from residual_policy import ResidualPolicy  # noqa: E402


def _latest_residual_ckpt(ckpt_dir: Path) -> Path | None:
    latest = ckpt_dir / "residual_latest.pt"  # the only file learner.py ever writes
    return latest if latest.exists() else None


def run_actor(checkpoint, icm_checkpoint, buffer_dir: Path, residual_ckpt_dir: Path, *,
              icm_filename: str = "icm_0708_proprio.pt", commit: bool, home: bool = True,
              force_limit_n: float | None = None, max_ticks: int = 1000,
              enforce_box: bool = False, explore: bool = True, layers: int | None = None,
              time_penalty: float = 0.01, force_soft_limit_n: float = 5.0,
              force_penalty_per_n: float = 0.02):
    mode = "GO — COMMANDS THE LEFT ARM" if commit else "DRY-RUN — commands nothing"
    print(f"{mode}. residual AWR actor | checkpoint: {checkpoint}\nicm: {icm_checkpoint}")
    from dexcontrol.robot import Robot
    from dexcontrol.core.config import get_robot_config
    from LGES.ik_demo import config as ikcfg
    from LGES.ik_demo.suction import SuctionMover
    from LGES.ik_demo.drivers import suction_io
    from LGES.ik_demo.go_home import both_arms_home
    from LGES.ik_demo.chassis_sequence import detect, _center_from_det, _view_park, set_head_pitch

    task = "case_pick"
    spec = TASKS[task]
    instruction, kind = spec["instruction"], spec["kind"]

    ob = ObsBuilder()
    policy, pre, post = load_policy(checkpoint)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    abs_action = int(policy.config.action_feature.shape[0]) == 8
    if abs_action:
        raise SystemExit("actor.py only supports delta-action checkpoints (residual is a "
                          "pose-delta correction) -- got an abs-action policy.")
    chunk_steps = int(getattr(policy.config, "n_action_steps", 1))
    layers = int(ikcfg.SRC_LAYERS_REMAINING if layers is None else layers)

    rewarder = ICMRewarder(icm_checkpoint, filename=icm_filename)
    print(f"reward shaping: -{time_penalty:g}/tick | "
          f"-{force_penalty_per_n:g}/N of contact over +{force_soft_limit_n:g}N")
    residual = ResidualPolicy(obs_dim=15)
    residual.eval()
    residual_ckpt_dir.mkdir(parents=True, exist_ok=True)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    writer = EpisodeWriter(buffer_dir)

    robot_configs = get_robot_config()
    robot_configs.enable_sensor("head_camera")
    robot_configs.sensors["head_camera"].transport = "zenoh"
    flim = ikcfg.FORCE_HARD_LIMIT_N if force_limit_n is None else force_limit_n
    with Robot(configs=robot_configs) as bot, SuctionMover(bot) as mover:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("  (head camera may not be active)")
        set_head_pitch(bot, angle=30.0)

        release = mover.software_estop_active()
        if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
            return
        if not mover.ensure_ready(release_estop=release):
            print("arm not ready — aborting")
            return

        if commit:
            print(f"\n  *** GO COMMANDS THE LEFT ARM. Keep a hand on the e-stop. ***\n"
                  f"  ENTER mid-episode = intervene (truncate, retreat-to-hover, NOT a reward) | "
                  f"force abort +{flim:.0f}N | <= {max_ticks} ticks/episode | "
                  f"runs continuously — Ctrl-C to stop")
            try:
                input("  >>> ENTER to authorize motion (Ctrl-C to cancel) <<< ")
            except (KeyboardInterrupt, EOFError):
                print("cancelled — no motion."); return

        seal = None
        try:
            seal = suction_io.VacuumMonitor(); seal.start()
        except Exception as e:  # noqa: BLE001
            print(f"  (vacuum monitor unavailable: {e}; seal -> 0)")

        dt = 1.0 / FPS
        run_num = 0
        try:
          while True:
            run_num += 1
            print(f"\n{'='*20} episode {run_num} {'='*20}")

            if home:
                both_arms_home(bot, left=mover)
            _view_park(mover, "policy")

            det = detect(bot, layers)
            box = None
            if det is not None and det.found:
                center = _center_from_det(det)
                box = _box_from_detection(center)
                print(f"  case @ xy=({center[0]:.3f},{center[1]:+.3f}) box {box[0].round(2)}..{box[1].round(2)}")
            elif enforce_box:
                print("  no detection — --box requires one; skipping this episode.")
                continue

            ckpt = _latest_residual_ckpt(residual_ckpt_dir)
            if ckpt is not None:
                residual = ResidualPolicy.load(ckpt)
                residual.eval()
                print(f"  residual: loaded {ckpt.name}")
            else:
                print("  residual: no checkpoint yet, using freshly-initialized (near-zero) residual")

            policy.reset()
            writer.open_episode(meta={"task": task, "instruction": instruction,
                                       "checkpoint": str(checkpoint), "residual_ckpt": str(ckpt)})
            baseline_f = _baseline_force(mover)
            entry_sealed = bool(seal.is_sealed()) if seal else False
            has_sealed, went_low = entry_sealed, False
            ref_pos = ref_quat = None
            prev_state = prev_action_exec = prev_action_residual = None
            task_done = run_stop = None
            max_tgt_z = -np.inf
            state = None

            try:
                for tick in range(max_ticks):
                    t0 = time.time()
                    torso_q = bot.torso.get_joint_pos()
                    left_q = np.asarray(bot.left_arm.get_joint_pos(), dtype=float)
                    right_q = bot.right_arm.get_joint_pos()
                    ws = getattr(mover._arm, "wrench_sensor", None)
                    wrench6 = (np.asarray(ws.get_state()["wrench"], float)[:6]
                               if ws is not None else np.zeros(6))
                    rgb, depth_m = _grab_rgb(bot), _grab_depth(bot)
                    suction = suction_io.is_suction_commanded_on()
                    sealed = bool(seal.is_sealed()) if seal else False
                    state = ob.state(torso_q, left_q, right_q, wrench6, suction, sealed)
                    # Correct the F/T baseline drift ONCE, at the source: every consumer
                    # of `state` -- the base policy's predict(), the residual policy, the
                    # ICM reward, and the buffer rows the learner trains on -- was
                    # trained/calibrated on the 0708 demos' wrench baseline, so all of
                    # them must see training-frame wrench values. The raw wrench6 stays
                    # untouched below for the live `contact` safety check, which tares
                    # itself against _baseline_force at episode start.
                    state[9:15] -= WRENCH_OFFSET_0716
                    cur_pos, cur_quat = state[:3], state[3:7]
                    contact = float(np.linalg.norm(wrench6[:3])) - baseline_f

                    # Close out the PREVIOUS tick's transition now that we have its
                    # next_state (this tick's `state`) -- see replay_buffer.py's
                    # docstring for why the bootstrap state must be exact, not off-by-one.
                    if prev_state is not None:
                        # No wrench_offset here: both states were already corrected above.
                        # Shaping on top of the ICM term: a flat per-tick time cost (the
                        # ICM reward is always positive, so without it dawdling
                        # on-manifold out-earns finishing), and a soft force penalty that
                        # grades overshoot BEFORE the hard +flim abort kills the episode.
                        # THIS tick's contact is the measured consequence of prev_action,
                        # so it belongs to the transition being closed here.
                        reward = (rewarder.reward(prev_state, prev_action_exec, state)
                                  - time_penalty
                                  - force_penalty_per_n * max(0.0, contact - force_soft_limit_n))
                        writer.step(prev_state, prev_action_residual, prev_action_exec, reward)

                    has_sealed = has_sealed or sealed
                    went_low = went_low or cur_pos[2] < DESCEND_Z
                    at_hover = cur_pos[2] >= HOVER_Z

                    # Done-check uses only THIS tick's already-computed state -- no new
                    # action is taken once done, so `state` doubles as the episode's
                    # terminal observation (and, for a truncation, the bootstrap state).
                    if kind == "pick" and has_sealed and went_low and at_hover:
                        task_done = f"grasped+lifted@{tick}"
                        break

                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.readline()
                        run_stop = "operator intervene (ENTER)"
                        break

                    pred = predict(policy, pre, post, state, ObsBuilder.image(rgb),
                                    instruction, ObsBuilder.depth_image(depth_m))
                    obs_t = torch.from_numpy(state).unsqueeze(0)
                    residual_raw = residual.act(obs_t, deterministic=not explore).squeeze(0).numpy()
                    combined = pred.copy()
                    combined[0:6] += residual_raw
                    clamped = clamp_action(combined)

                    if tick % chunk_steps == 0:
                        ref_pos, ref_quat = cur_pos.copy(), cur_quat.copy()
                    pos_tgt, rpy_tgt, _R_tgt = integrate(ref_pos, ref_quat, clamped)
                    ref_pos, ref_quat = pos_tgt, _rpy_to_quat_wxyz(rpy_tgt)
                    max_tgt_z = max(max_tgt_z, float(pos_tgt[2]))

                    in_box = workspace_ok(pos_tgt, box) if box is not None else True
                    sol = mover.solve_pose(pos_tgt, rpy_tgt, seed=left_q, min_motion=True)
                    dq = float(np.abs(sol.q - left_q).max())

                    clip = "CLIP" if not np.allclose(clamped, combined) else "    "
                    # \033[K (clear-to-end-of-line) after the content, not just a bare
                    # \r: without it, a shorter line doesn't erase the previous longer
                    # line's tail, so old digits visually bleed into the new ones.
                    print(f"\r[{tick:3d}] dpos={clamped[:3]*1000} suc={clamped[6]:.2f} {clip} "
                          f"z={cur_pos[2]:.2f} contact={contact:+.1f}N seal={'Y' if sealed else '.'}"
                          "\033[K", end="", flush=True)

                    if commit:
                        if contact > flim:
                            run_stop = f"ABORT: force {contact:.1f}N > +{flim:.1f}N"; break
                        if enforce_box and box is not None and not in_box:
                            run_stop = "ABORT: target out of the detection box"; break
                        if dq > MAX_JOINT_STEP_RAD:
                            run_stop = f"ABORT: joint jump {dq:.2f} rad (IK near-singular?)"; break
                        mover._arm.set_joint_pos(sol.q)
                        want_suction = clamped[6] > 0.5
                        if want_suction != suction_io.is_suction_commanded_on():
                            (suction_io.suction_on if want_suction else suction_io.suction_off)()

                    prev_state = state
                    prev_action_exec = clamped.copy()
                    prev_action_residual = residual_raw.copy()

                    time.sleep(max(0.0, dt - (time.time() - t0)))
                else:
                    run_stop = f"STALL: did not finish in {max_ticks} ticks"
            except KeyboardInterrupt:
                run_stop = "ABORT: KeyboardInterrupt"
            finally:
                writer.close(terminal=(task_done is not None),
                             last_next_state=None if task_done is not None else state)

            if commit:
                # release(), not a bare suction_off(): the DI0 seal signal stays "sealed"
                # for a while AFTER suction is commanded off until the cup actually lets
                # go (see suction_io.py's VacuumMonitor docstring) -- moving immediately
                # (the next episode's both_arms_home is a big, non-vertical motion) drags
                # a still-attached case sideways instead of dropping it in place.
                if has_sealed or suction_io.is_suction_commanded_on():
                    suction_io.release()
                else:
                    suction_io.suction_off()
                if task_done is None:
                    try:
                        cap = max_tgt_z if np.isfinite(max_tgt_z) else None
                        _retreat_to_hover(mover, cap_z=cap)
                    except KeyboardInterrupt:
                        print("\n  retreat interrupted by operator")
                    except Exception as e:  # noqa: BLE001
                        print(f"\n  retreat failed ({e}); leaving arm in place")

            print(f"\nepisode ended: {task_done or run_stop}")
            if run_stop == "ABORT: KeyboardInterrupt":
                break
            # Runs continuously -- no per-episode prompt: the whole point of this actor
            # is unattended online data collection, with ENTER mid-episode as the only
            # human touchpoint (intervene/truncate, see the module docstring). Ctrl-C
            # (caught above as run_stop) is the only way to stop between episodes.
        finally:
            if seal is not None:
                seal.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="Chanho-Lee/smolvla_meanflow_naive_0708",
                    help="HF repo id or local path -- default is the meanflow checkpoint "
                         "(its 1-step generation is what makes online rollouts/relabeling "
                         "cheap; see its README's 'Notes for RL fine-tuning afterwards')")
    ap.add_argument("--icm-checkpoint", default="Chanho-Lee/icm_case_pick_0708",
                    help="HF repo id or local path/dir containing --icm-filename")
    ap.add_argument("--icm-filename", default="icm_0708_proprio.pt",
                    help="file inside --icm-checkpoint when it's a repo id or dir")
    ap.add_argument("--buffer-dir", type=Path, default=HERE / "rl_buffer" / "case_pick",
                    help="episode-shard replay buffer dir (gitignored; default under this dir)")
    ap.add_argument("--residual-ckpt-dir", type=Path, default=HERE / "rl_ckpt" / "case_pick",
                    help="learner.py's checkpoint dir, polled between episodes (gitignored; default under this dir)")
    ap.add_argument("--dry-run", action="store_true", help="no robot motion; exercises the reward/buffer plumbing")
    ap.add_argument("--go", action="store_true", help="COMMANDS the left arm + suction")
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--box", action="store_true")
    ap.add_argument("--force-limit", type=float, default=20.0)
    ap.add_argument("--time-penalty", type=float, default=0.01,
                    help="flat per-tick reward penalty -- the ICM term is always positive, "
                         "so without this, dawdling on-manifold out-earns finishing (0 disables)")
    ap.add_argument("--force-soft-limit", type=float, default=5.0,
                    help="contact force (N over baseline) where the soft penalty starts; "
                         "keep well under --force-limit so the residual learns to back off "
                         "before the hard abort")
    ap.add_argument("--force-penalty", type=float, default=0.02,
                    help="reward penalty per N of contact above --force-soft-limit (0 disables)")
    ap.add_argument("--max-ticks", type=int, default=1000)
    ap.add_argument("--no-explore", action="store_true", help="use the residual's deterministic mean (eval, not data collection)")
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()

    if not (args.dry_run or args.go):
        ap.error("choose --dry-run or --go")

    run_actor(args.checkpoint or latest_checkpoint(), args.icm_checkpoint, args.buffer_dir,
              args.residual_ckpt_dir, icm_filename=args.icm_filename, commit=args.go,
              home=not args.no_home, force_limit_n=args.force_limit, max_ticks=args.max_ticks,
              enforce_box=args.box, explore=not args.no_explore, layers=args.layers,
              time_penalty=args.time_penalty, force_soft_limit_n=args.force_soft_limit,
              force_penalty_per_n=args.force_penalty)


if __name__ == "__main__":
    main()
