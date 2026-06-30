#!/usr/bin/env python3
"""Human-in-the-loop intervention executor (HG-DAgger).  See INTERVENTION_DESIGN.md.

Runs the policy live (reusing LGES/vla_training/run_policy machinery) with keyboard
takeover: press TAB to seize control and jog the EE (teach_pose keys), TAB again to
hand back. Human-controlled segments are recorded as standard takes (RecordController)
for DAgger retraining; convert_to_lerobot / convert_prevaction ingest them unchanged.

Reuses run_policy (load/obs/clamp/integrate/IK/Mover/goto-start/retreat/safety),
teach_pose (cbreak keys + EE-jog map) and dashboard recorder/publisher. Modifies none.

Keys:  TAB toggle policy<->human | (human) w/s a/d r/f = +-xyz, u/o i/k j/l = rpy,
       x = toggle suction, +/- = jog step | g = reset to jittered start |
       n = mark last takeover FAILED | q/Ctrl-C = stop

Run with the vla_venv python (case PRESENT, hand on the e-stop; NOT while training
shares the GPU):
  /home/dexmate/vla_venv/bin/python Research/intervention/intervene.py \
      --checkpoint Research/contact_aware_vla/outputs/smolvla_prevaction_0617/checkpoints/last \
      --task case_pick --goto-start LGES/recordings/case_pick/<take> --force-limit 15
"""

import argparse
import json
import sys
import termios
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VLA_TRAIN = ROOT / "LGES" / "vla_training"
sys.path.insert(0, str(VLA_TRAIN))

import run_policy as rp  # noqa: E402
from run_policy import (  # noqa: E402
    load_policy, ObsBuilder, clamp_action, integrate, workspace_ok, _to_chw,
    _goto_start, _baseline_force, _retreat_to_hover, _grab_rgb, _grab_depth,
    _rpy_to_quat_wxyz, TASKS, SUCTION_SEQUENCE, FPS, MAX_JOINT_STEP_RAD,
    HOVER_Z, DESCEND_Z)
from convert_to_lerobot import quat_mul, quat_conj, quat_to_rotvec  # noqa: E402
from case_battery_demo.teach_pose import _POS_KEYS, _RPY_KEYS, _get_key  # noqa: E402

# ── shared keyboard state (GIL-atomic flags; jog accumulated under a lock) ──
_lock = threading.Lock()
ctl = {"mode": "POLICY", "suction": False, "step_m": 0.005, "ostep": np.deg2rad(5),
       "djog": np.zeros(6), "fail": False, "abort": False, "reset": False,
       "manipulated": False, "go": False}


def _keyboard_thread():
    """Owns stdin (cbreak via teach_pose._get_key); updates `ctl`. The MAIN loop
    acts on the flags — no robot/policy calls happen here."""
    while not ctl["abort"]:
        try:
            k = _get_key()
        except Exception:  # noqa: BLE001
            return
        if k == "q" or (len(k) and ord(k) == 3):          # q / Ctrl-C
            ctl["abort"] = True; return
        if k in ("\r", "\n"):                              # ENTER: start/resume the attempt
            ctl["go"] = True
        elif k == "\t":                                    # TAB: toggle takeover
            ctl["mode"] = "HUMAN" if ctl["mode"] == "POLICY" else "POLICY"
        elif k == "x":
            ctl["suction"] = not ctl["suction"]; ctl["manipulated"] = True
        elif k == "n":
            ctl["fail"] = True
        elif k == "g":
            ctl["reset"] = True
        elif k in ("+", "="):
            ctl["step_m"] = min(ctl["step_m"] * 2, 0.05)
        elif k == "-":
            ctl["step_m"] = max(ctl["step_m"] / 2, 0.001)
        elif k in _POS_KEYS:
            ax, sg = _POS_KEYS[k]
            ctl["manipulated"] = True
            with _lock:
                ctl["djog"][ax] += sg * ctl["step_m"]
        elif k in _RPY_KEYS:
            ax, sg = _RPY_KEYS[k]
            ctl["manipulated"] = True
            with _lock:
                ctl["djog"][3 + ax] += sg * ctl["ostep"]


def _policy_pred(policy, pre, post, state, img, depth, instruction, prev_action):
    """Run the policy; state is 15-dim, prev_action (7) appended iff the model wants 22."""
    s = state if prev_action is None else np.concatenate([state, prev_action]).astype(np.float32)
    obs = {"observation.images.head": _to_chw(img),
           "observation.state": torch.from_numpy(s).unsqueeze(0),
           "task": instruction,
           "observation.images.head_depth": _to_chw(depth)}
    obs = pre(obs)
    with torch.inference_mode():
        return post(policy.select_action(obs)).squeeze(0).cpu().numpy()


def _read_start_pose(take_dir):
    """First-frame EE pose (pos, rpy) of a recorded take — the start pose."""
    f0 = json.loads((take_dir / "states.jsonl").open().readline())
    pos = np.asarray(f0["ee"]["pos"], float)
    w, x, y, z = f0["ee"]["quat_wxyz"]
    return pos, Rotation.from_quat([x, y, z, w]).as_euler("xyz")


def _goto_jittered(mover, start_pos, start_rpy, jitter, rng):
    """Move to the start pose + uniform xy jitter (varied starts for DAgger)."""
    jit = np.zeros(3)
    if jitter > 0:
        jit[:2] = rng.uniform(-jitter, jitter, size=2)
    tgt = np.asarray(start_pos, float) + jit
    print(f"\n  -> goto start {tgt.round(3)} (xy jitter {jit[:2].round(3)})")
    mover.goto(tgt, start_rpy, step_duration=3.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--task", default="case_pick", choices=list(TASKS))
    ap.add_argument("--chain", nargs="+", default=None,
                    help="run a sequence, e.g. --chain case_pick case_place (or 'all'); overrides --task")
    ap.add_argument("--goto-start", type=Path, default=None)
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--box", action="store_true", help="enforce the per-task workspace box")
    ap.add_argument("--force-limit", type=float, default=None)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--jitter", type=float, default=0.02,
                    help="xy jitter (m) applied to the start pose on the 'g' reset key")
    ap.add_argument("--vertical", action="store_true", default=True,
                    help="lock commanded EE orientation to the demo's cup-down start orientation; position-only")
    ap.add_argument("--prev-action", choices=["auto", "on", "off"], default="on",
                    help="feed previous action (22-dim). auto = infer from checkpoint name")
    ap.add_argument("--record-dir", type=Path, default=HERE / "interventions")
    args = ap.parse_args()

    use_prev = (args.prev_action == "on" or
                (args.prev_action == "auto" and "prevaction" in str(args.checkpoint).lower()))
    spec = TASKS[args.task]
    instruction, box = spec["instruction"], spec["box"]

    RECORDINGS = VLA_TRAIN.parent / "recordings"
    if args.goto_start is None:                      # default to the task's first demo take
        _takes = sorted(p for p in (RECORDINGS / args.task).iterdir() if p.is_dir())
        if not _takes:
            sys.exit(f"no takes under {RECORDINGS / args.task} to default --goto-start")
        args.goto_start = _takes[0]
        print(f"--goto-start not given; defaulting to first take: {args.goto_start.name}")
    start_pos, start_rpy = _read_start_pose(args.goto_start)
    rng = np.random.default_rng()

    from dexcontrol.robot import Robot
    from dexcontrol.core.config import get_robot_config
    from case_battery_demo.grasp import SuctionMover
    from case_battery_demo.home_pose import go_to_default_pose
    from case_battery_demo import suction_io, config as cfg
    from case_battery_demo.dashboard.recorder import RecordController
    from case_battery_demo.dashboard.publisher import DashboardPublisher

    ob = ObsBuilder()
    policy, pre, post = load_policy(args.checkpoint)
    chunk_steps = int(getattr(policy.config, "n_action_steps", 1))
    flim = cfg.FORCE_HARD_LIMIT_N if args.force_limit is None else args.force_limit
    # "vertical" = the demo's actual cup-down orientation (from the start take), NOT
    # cfg.GRASP_ORIENTATION_RPY — the latter differs ~110deg in yaw from the recorded
    # demo/policy pose, so forcing it would snap the wrist and trip the joint-jump guard.
    vertical_rpy = np.asarray(start_rpy, float) if args.vertical else None
    print(f"checkpoint: {args.checkpoint}\ntask: {args.task} | prev_action={use_prev} | "
          f"force-limit +{flim:.0f}N | box {'ON' if args.box else 'off'} | "
          f"vertical {'ON' if args.vertical else 'off'}")

    robot_configs = get_robot_config()
    robot_configs.enable_sensor("head_camera")
    robot_configs.sensors["head_camera"].transport = "zenoh"

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    with Robot(configs=robot_configs) as bot, SuctionMover(bot) as mover:
        if not args.no_home:
            go_to_default_pose(bot)
        _goto_start(mover, args.goto_start)          # exact demo start (goto_start defaulted above)
        # head tilt to see the workspace (same source run_policy/run_demo use)
        import importlib.util
        sp = importlib.util.spec_from_file_location("lges_utils", VLA_TRAIN.parent / "utils.py")
        u = importlib.util.module_from_spec(sp); sp.loader.exec_module(u)
        u.set_head_pitch(bot, pitch_deg=30.0)

        seal = None
        try:
            seal = suction_io.VacuumMonitor(); seal.start()
        except Exception as e:  # noqa: BLE001
            print(f"  (vacuum monitor unavailable: {e})")

        args.record_dir.mkdir(parents=True, exist_ok=True)
        recorder = RecordController(out_dir=str(args.record_dir),
                                    spool_dir=str(args.record_dir / ".spool"),
                                    instruction=instruction)
        recorder.set_meta_extra({"intervention": True})   # per-take task+instruction set at episode_begin
        recorder.start()
        publisher = DashboardPublisher(bot, on_sample=recorder.feed).start()

        threading.Thread(target=_keyboard_thread, daemon=True).start()
        print("\n  *** ENTER = start attempt.  TAB = take over / hand back.  "
              "g = reset to jittered start.  q = stop.  Hand on the e-stop. ***\n")

        dt = 1.0 / FPS
        tasks = list(SUCTION_SEQUENCE) if args.chain == ["all"] else (args.chain or [args.task])
        bad = [t for t in tasks if t not in TASKS]
        if bad:
            sys.exit(f"unknown task(s) {bad}; choices {list(TASKS)} (or 'all')")

        ref_pos = ref_quat = None          # policy integration reference
        target_pos = target_rpy = None     # current EE setpoint
        prev_pos = prev_quat = None        # for the realized prev-action delta
        prev_mode = "POLICY"
        recording_started = False          # episode_begin deferred to first manipulation
        stop = None

        def _spec(i):
            t = tasks[i]; s = TASKS[t]
            return t, s["instruction"], s["box"], s["kind"]

        def _task_latches():                 # done-detection state at a task's start
            es = bool(seal.is_sealed()) if seal else False
            return es, es, False, False      # entry_sealed, has_sealed, has_released, went_low

        task_idx = 0
        task, instruction, box, kind = _spec(0)
        baseline_f = _baseline_force(mover)
        entry_sealed, has_sealed, has_released, went_low = _task_latches()
        ctl["go"] = False                    # wait for ENTER to start the first attempt
        print(f"\n=== chain: {' -> '.join(tasks)} ===")
        print(f"  press ENTER to start '{task}'  (g = reset to jittered start, q = quit)")

        tick = 0
        try:
            while not ctl["abort"] and tick < args.max_ticks:
                t0 = time.time()

                # 'g': restart the chain from a fresh jittered start, then wait for ENTER
                if ctl["reset"]:
                    ctl["reset"] = False
                    if recording_started:
                        recorder.episode_end(success=not ctl["fail"]); recording_started = False
                    ctl["fail"] = False; ctl["mode"] = prev_mode = "POLICY"
                    suction_io.suction_off()
                    _goto_jittered(mover, start_pos, start_rpy, args.jitter, rng)
                    task_idx = 0; task, instruction, box, kind = _spec(0)
                    policy.reset(); ref_pos = ref_quat = None; prev_pos = prev_quat = None
                    baseline_f = _baseline_force(mover)
                    entry_sealed, has_sealed, has_released, went_low = _task_latches()
                    ctl["go"] = False; tick = 0
                    print(f"\n  reset to jittered start; chain from '{task}'. press ENTER to start.")
                    continue

                # ENTER-gate: hold (arm stays put) until the operator presses ENTER
                if not ctl["go"]:
                    time.sleep(0.05); continue

                torso_q = bot.torso.get_joint_pos()
                left_q = bot.left_arm.get_joint_pos()
                right_q = bot.right_arm.get_joint_pos()
                ws = getattr(mover._arm, "wrench_sensor", None)
                wrench6 = (np.asarray(ws.get_state()["wrench"], float)[:6]
                           if ws is not None else np.zeros(6))
                rgb, depth_m = _grab_rgb(bot), _grab_depth(bot)
                suction = suction_io.is_suction_commanded_on()
                sealed = bool(seal.is_sealed()) if seal else False
                state = ob.state(torso_q, left_q, right_q, wrench6, suction, sealed)
                cur_pos, cur_quat = state[:3].copy(), state[3:7].copy()

                # realized previous-action delta (matches convert_prevaction)
                if use_prev:
                    if prev_pos is None:
                        prev_action = np.zeros(7, dtype=np.float32)
                    else:
                        drot = quat_to_rotvec(quat_mul(cur_quat, quat_conj(prev_quat)))
                        prev_action = np.concatenate(
                            [cur_pos - prev_pos, drot, [float(state[7])]]).astype(np.float32)
                else:
                    prev_action = None
                prev_pos, prev_quat = cur_pos, cur_quat

                mode = ctl["mode"]
                if mode != prev_mode:                    # ── handoff ──
                    if mode == "HUMAN":
                        target_pos, target_rpy = mover.current_ee_pose()
                        target_rpy = np.array(target_rpy, float)
                        ctl["suction"] = suction
                        ctl["manipulated"] = False; recording_started = False
                        print(f"\n[{task} {tick}] >>> HUMAN takeover — recording starts on first jog <<<")
                    else:
                        if recording_started:
                            recorder.episode_end(success=not ctl["fail"]); recording_started = False
                        ctl["fail"] = False
                        policy.reset()                   # clear the stale 50-step chunk
                        ref_pos = ref_quat = None        # re-ground integration
                        baseline_f = _baseline_force(mover)  # payload may have changed
                        print(f"[{task} {tick}] <<< handed back to POLICY (chunk reset) >>>")
                    prev_mode = mode

                if mode == "POLICY":
                    pred = _policy_pred(policy, pre, post, state, ObsBuilder.image(rgb),
                                        ObsBuilder.depth_image(depth_m), instruction, prev_action)
                    clamped = clamp_action(pred)
                    if ref_pos is None or tick % chunk_steps == 0:
                        ref_pos, ref_quat = cur_pos.copy(), cur_quat.copy()
                    target_pos, target_rpy, _ = integrate(ref_pos, ref_quat, clamped)
                    ref_pos, ref_quat = target_pos, _rpy_to_quat_wxyz(target_rpy)
                    want_suction = pred[6] > 0.5
                else:  # HUMAN: apply accumulated jog to the setpoint
                    with _lock:
                        dj = ctl["djog"].copy(); ctl["djog"][:] = 0.0
                    target_pos = target_pos + dj[:3]
                    target_rpy = target_rpy + dj[3:]
                    want_suction = ctl["suction"]
                    if not recording_started and ctl["manipulated"]:  # record from first manipulation
                        recorder.set_meta_extra({"intervention": True, "task": task,
                                                 "instruction": instruction})
                        recorder.episode_begin(f"intervention_{task}")
                        time.sleep(0.2)                  # let the recorder flip on
                        recording_started = True
                        print(f"\n[{task} {tick}] >>> recording (manipulation started) <<<")

                if vertical_rpy is not None:         # lock orientation to vertical (position-only)
                    target_rpy = vertical_rpy

                in_box = workspace_ok(target_pos, box)
                q, _ = mover._solve_ik(mover._fresh_configuration(), target_pos, target_rpy)
                arm_q_cmd = mover._arm_joints_from_q(q)
                dq = float(np.abs(arm_q_cmd - left_q).max())
                contact = float(np.linalg.norm(wrench6[:3])) - baseline_f

                if dq > MAX_JOINT_STEP_RAD:
                    stop = f"ABORT: joint jump {dq:.2f} rad (IK near-singular)"; break
                if contact > flim:
                    stop = f"ABORT: force {contact:.1f}N > +{flim:.0f}N"; break
                if args.box and not in_box:
                    stop = f"ABORT: target {target_pos.round(3)} out of box"; break

                mover._arm.set_joint_pos(arm_q_cmd)
                if want_suction != suction_io.is_suction_commanded_on():
                    (suction_io.suction_on if want_suction else suction_io.suction_off)()

                # task done-detection (same conditions as run_policy) + advance the chain
                has_sealed = has_sealed or sealed
                went_low = went_low or cur_pos[2] < DESCEND_Z
                if entry_sealed and not sealed:
                    has_released = True
                at_hover = cur_pos[2] >= HOVER_Z
                done = ((kind == "pick" and has_sealed and went_low and at_hover) or
                        (kind == "place" and has_released and went_low and at_hover))

                print(f"[{kind[0].upper()} {task} {tick:4d} {mode[0]}] z={cur_pos[2]:.3f} "
                      f"box={'ok' if in_box else 'OUT'} contact={contact:+.1f}N "
                      f"seal={'Y' if sealed else '.'} suc={'on' if want_suction else 'off'}", end="\r")

                if done:
                    if recording_started:
                        recorder.episode_end(success=not ctl["fail"])
                        recording_started = False; ctl["fail"] = False
                    task_idx += 1
                    if task_idx >= len(tasks):
                        print(f"\n  *** chain complete ({' -> '.join(tasks)}). g = another round, q = quit. ***")
                        ctl["go"] = False
                        continue
                    task, instruction, box, kind = _spec(task_idx)
                    policy.reset(); ref_pos = ref_quat = None
                    baseline_f = _baseline_force(mover)
                    entry_sealed, has_sealed, has_released, went_low = _task_latches()
                    tick = 0
                    print(f"\n  -> next task: '{task}'  | \"{instruction}\"")
                    continue

                tick += 1
                time.sleep(max(0.0, dt - (time.time() - t0)))

            if stop is None:
                stop = "operator stop (q)" if ctl["abort"] else "max active ticks reached"
        except KeyboardInterrupt:
            stop = "KeyboardInterrupt"
        finally:
            if recording_started:                    # close an open intervention take
                recorder.episode_end(success=not ctl["fail"])
            try:
                suction_io.suction_off()
                _retreat_to_hover(mover)
            except Exception as e:  # noqa: BLE001
                print(f"\n  retreat failed ({e})")
            publisher.stop(); recorder.stop()
            if seal:
                seal.stop()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        print(f"\nended: {stop}. interventions saved under {args.record_dir}")


if __name__ == "__main__":
    main()


#/home/dexmate/vla_venv/bin/python Research/intervention/intervene.py --checkpoint LGES/vla_training/outputs/smolvla_baseline_0617/checkpoints/last --force-limit 20 --chain case_pick case_place