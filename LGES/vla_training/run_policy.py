#!/usr/bin/env python3
"""On-robot executor for the trained SmolVLA policy.

Per EXECUTOR_DESIGN.md. Scope: one task (case_pick), left arm, suction.
Three modes:

  --self-test TAKE_DIR   offline, no robot. Rebuilds the 14-dim observation
                         from a recorded take's joints (the same path the
                         live loop uses) and checks it reproduces the take's
                         stored state, then runs the policy and prints
                         predicted-vs-recorded actions. Validates obs parity
                         and the delta-integration math (design open Q1/Q2/Q4).

  --dry-run              live, needs the robot. Loops at ~15 Hz: reads live
                         observation, runs the policy, prints predicted action
                         + safety-clamped target + IK feasibility. Commands
                         NOTHING.

  --go                   live. Same loop but COMMANDS the left arm (IK ->
                         set_joint_pos) and suction, fully guarded (per-step
                         clamp, workspace box, force limit, IK-fail, operator
                         abort) and stopping on vacuum seal. Requires
                         --goto-start so the arm begins in-distribution.

Run with the vla_venv python (it can import lerobot AND the demo stack):
  /home/dexmate/vla_venv/bin/python run_policy.py --self-test <take_dir>
  /home/dexmate/vla_venv/bin/python run_policy.py --dry-run --goto-start <take_dir>
  /home/dexmate/vla_venv/bin/python run_policy.py --go --goto-start <take_dir>
"""

import argparse
import json
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

VLA_DIR = Path(__file__).resolve().parent
LGES_DIR = VLA_DIR.parent
# publisher/grasp use package-relative imports (from .. import config), so the
# demo must be importable as the `case_battery_demo` package -> LGES on path.
sys.path.insert(0, str(LGES_DIR))
sys.path.insert(0, str(LGES_DIR / "case_battery_demo"))  # grasp's own `import config`
sys.path.insert(0, str(VLA_DIR))  # convert_to_lerobot (shared colorize_depth)

# Reuse the converter's exact depth colorize so live obs == training data.
from convert_to_lerobot import colorize_depth  # noqa: E402

IMG_W, IMG_H = 512, 320  # must match convert_to_lerobot.py
FPS = 15

# Per-step safety clamps. These bound a single step but must NOT throttle
# normal motion: at half these values the arm just hovered (real predicted
# deltas got clipped and never reached the case). [10,50,50] mm clears a full
# case_pick while staying within the training action range (dx max ~11.6 mm,
# dy/dz under their maxima), so it still catches out-of-distribution spikes.
MAX_DPOS_M = np.array([0.01, 0.03, 0.03])
MAX_DROT_RAD = 0.025

# A sub-task isn't done at the grasp/release instant — the recorded episodes
# end AFTER the arm lifts the case back to the ~1.09 m hover (and a place ends
# after retracting to hover). So a task completes only once it has sealed (pick)
# / released (place) AND the EE has returned to hover height. This also keeps
# the next task starting in-distribution (every task begins at this hover). All
# recorded task end-poses sit at z~1.09; 1.05 gives margin while clearly above
# the contact depth (~0.78). DESCEND_Z: a task only counts as done once the arm
# has actually gone down to the object (z below this) — guards against a seal
# flicker at the hover start of a place falsely completing it at tick 0.
HOVER_Z = 1.05
DESCEND_Z = 1.00

# IK non-convergence is NOT a failure — the pink solver returns a best-effort
# joint config and the demo commands it regardless (grasp.py discards the ok
# flag). The real danger is a genuine blow-up near a singularity: a large
# one-tick joint jump. EE deltas are clamped to mm-scale, so a normal tick
# moves each joint well under this; a jump above it means trouble -> abort.
MAX_JOINT_STEP_RAD = 1.2

# Per-task config: instruction the policy trained on, kind (pick ends on
# vacuum seal / place ends on release), and a workspace box = the recorded EE
# range for that task + ~5 cm margin (base_link m). Hard-stop outside the box.
# Boxes are per-task on purpose: carrying a payload near the slots, a tight box
# is the main guard against the policy driving to the wrong region.
TASKS = {
    "case_pick":       dict(instruction="pick up the case with the suction cup",
                            kind="pick",  box=(np.array([0.64, 0.06, 0.73]), np.array([0.82, 0.60, 1.16]))),
    "case_place":      dict(instruction="place the case on the right workspace",
                            kind="place", box=(np.array([0.64, 0.05, 0.71]), np.array([0.85, 0.59, 1.16]))),
    "battery_1_pick":  dict(instruction="Pick up the right battery with the suction cup",
                            kind="pick",  box=(np.array([0.63, 0.05, 0.71]), np.array([0.84, 0.60, 1.16]))),
    "battery_1_place": dict(instruction="Insert the battery into right slot of the case",
                            kind="place", box=(np.array([0.64, -0.09, 0.72]), np.array([0.83, 0.60, 1.15]))),
    "battery_2_pick":  dict(instruction="Pick up the left battery with the suction cup",
                            kind="pick",  box=(np.array([0.62, -0.09, 0.71]), np.array([0.82, 0.60, 1.18]))),
    "battery_2_place": dict(instruction="Insert the battery into left slot of the case",
                            kind="place", box=(np.array([0.65, 0.05, 0.72]), np.array([0.83, 0.60, 1.16]))),
}
SUCTION_SEQUENCE = list(TASKS)


# ── observation builder (shared by self-test and live loop) ──────────


def _rpy_to_quat_wxyz(rpy) -> np.ndarray:
    """Exactly the recorder's conversion (recorder.py:_rpy_to_quat_wxyz)."""
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class ObsBuilder:
    """Reconstructs the state + head RGB + colorized head depth the policy
    expects (matches convert_to_lerobot.py exactly).

    Uses the publisher's _EEKinematics (the FK that produced the training
    `states.jsonl`) so live observations match the recordings. State layout:
    pos(3) quat_wxyz(4) suction(1) vacuum_sealed(1) raw-wrench fx..tz(6).
    """

    def __init__(self):
        from case_battery_demo.dashboard.publisher import _EEKinematics
        self._fk = _EEKinematics()

    def state(self, torso_q, left_q, right_q, wrench6, suction_on: bool,
              sealed: bool) -> np.ndarray:
        pos, rpy = self._fk.compute(torso_q, left_q, right_q)["left"]
        quat = _rpy_to_quat_wxyz(rpy)
        s = np.concatenate([
            np.asarray(pos, dtype=np.float64),
            quat,
            [1.0 if suction_on else 0.0],
            [1.0 if sealed else 0.0],
            np.asarray(wrench6, dtype=np.float64)[:6],
        ])
        return s.astype(np.float32)

    @staticmethod
    def image(rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> resized RGB, matching convert_to_lerobot.py."""
        import cv2
        # converter reads BGR jpgs, resizes, then BGR->RGB. Live frames are
        # already RGB, so resize directly (no colour swap) to land identical.
        return cv2.resize(rgb, (IMG_W, IMG_H))

    @staticmethod
    def depth_image(depth_m: np.ndarray) -> np.ndarray:
        """Live head depth (metres) -> colorized RGB == training camera2.
        Replicates the recorder's metres->uint16 mm, then the converter's
        colorize_depth, so deploy matches the dataset byte-for-byte."""
        d = np.asarray(depth_m, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        mm = np.clip(np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0, neginf=0.0),
                     0, 65535).astype(np.uint16)
        return colorize_depth(mm)


# ── action decode / integrate / clamp (shared) ───────────────────────


def clamp_action(act: np.ndarray) -> np.ndarray:
    out = act.copy()
    out[0:3] = np.clip(out[0:3], -MAX_DPOS_M, MAX_DPOS_M)
    out[3:6] = np.clip(out[3:6], -MAX_DROT_RAD, MAX_DROT_RAD)
    return out


def integrate(cur_pos, cur_quat_wxyz, act):
    """Apply a base-frame delta to the live pose. Returns (pos, rpy, R).

    Mirrors the converter: action drot = rotvec(q_{t+1} q_t^{-1}), a base-frame
    (left) rotation, so we LEFT-multiply: R_tgt = R_delta @ R_cur.
    """
    dpos, drot = act[0:3], act[3:6]
    w, x, y, z = cur_quat_wxyz
    R_cur = Rotation.from_quat([x, y, z, w]).as_matrix()
    R_delta = Rotation.from_rotvec(drot).as_matrix()
    R_tgt = R_delta @ R_cur
    pos_tgt = np.asarray(cur_pos) + dpos
    return pos_tgt, Rotation.from_matrix(R_tgt).as_euler("xyz"), R_tgt


def workspace_ok(pos, box) -> bool:
    lo, hi = box
    return bool(np.all(pos >= lo) and np.all(pos <= hi))


# ── policy wrapper ────────────────────────────────────────────────────


def load_policy(checkpoint: Path, film: bool = False):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    if film:
        # FiLM condition-conditioned policy (V1/V2): patch VLAFlowMatching BEFORE
        # from_pretrained so contact_film + buffers exist and load from the checkpoint.
        # c-hat is then computed live from the obs. Variant is train-time only, so 'v2' is
        # fine here. FILM_COND + FILM_MASK_FORCE MUST match what the checkpoint was trained with.
        import os
        import film_contact
        mask_force = os.environ.get("FILM_MASK_FORCE", "1") not in ("0", "false", "False")
        cond = tuple(c.strip() for c in os.environ.get("FILM_COND", "contact,fz,seal").split(",") if c.strip())
        inject = os.environ.get("FILM_INJECT", "suffix")
        f0 = float(os.environ.get("FILM_F0", "14"))
        tau = float(os.environ.get("FILM_TAU", "3"))         # contact-DROP scale; MUST match training
        fz_tau = float(os.environ.get("FILM_FZ_TAU", "30"))  # fz scale; MUST match training
        ds = VLA_DIR / "datasets/lges_suction"
        wm, ws = film_contact.load_wrench_stats(ds)
        sm, ss = film_contact.load_seal_stats(ds)
        film_contact.apply("v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
                           contact_F0=f0, contact_tau=tau, fz_tau=fz_tau,
                           mask_force=mask_force, inject=inject)
        print(f"[run_policy] FiLM ENABLED (cond={cond} inject={inject} mask_force={mask_force} "
              f"F0={f0:.0f} tau={tau:.0f} fz_tau={fz_tau:.0f})")
    model_dir = checkpoint / "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_dir)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    return policy, pre, post


def _to_chw(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def predict(policy, pre, post, state: np.ndarray, image: np.ndarray,
            instruction: str, depth_image: np.ndarray | None = None) -> np.ndarray:
    obs = {
        "observation.images.head": _to_chw(image),
        "observation.state": torch.from_numpy(state).unsqueeze(0),
        "task": instruction,
    }
    if depth_image is not None:
        obs["observation.images.head_depth"] = _to_chw(depth_image)
    obs = pre(obs)
    with torch.inference_mode():
        action = policy.select_action(obs)
    return post(action).squeeze(0).cpu().numpy()


def latest_checkpoint() -> Path:
    runs = sorted((VLA_DIR / "outputs").glob("*/checkpoints/last"))
    if not runs:
        raise SystemExit("no checkpoint under outputs/*/checkpoints/last")
    return runs[-1]


# ── self-test (offline, no robot) ─────────────────────────────────────


def self_test(take_dir: Path, checkpoint: Path):
    print(f"self-test on {take_dir.name}\ncheckpoint: {checkpoint}\n")
    instruction = json.loads((take_dir / "meta.json").read_text())["instruction"]
    frames = [json.loads(l) for l in (take_dir / "states.jsonl").open()]
    rgb_paths = sorted((take_dir / "head_rgb").glob("*.jpg"))
    depth_paths = sorted((take_dir / "head_depth").glob("*.png"))

    ob = ObsBuilder()

    # 1. obs parity: rebuild state from joints, compare to stored state.
    #    The recorder samples the joint dict and the EE-FK from two separate
    #    joint reads in one tick, so during motion the rebuilt EE can't match
    #    the logged EE to the micron. We therefore judge parity on STATIC
    #    frames (no inter-frame motion): there the rebuild must be exact, which
    #    proves the FK path is identical to the one that made the training data.
    print("== observation parity (rebuilt vs recorded) ==")
    static_pos = static_quat = 0.0
    moving_pos = 0.0
    for i, f in enumerate(frames):
        j = f["joints"]
        w = f["wrench"]
        wrench6 = [w["fx"], w["fy"], w["fz"], w["tx"], w["ty"], w["tz"]]
        s = ob.state(list(j["torso"].values()), list(j["left_arm"].values()),
                     list(j["right_arm"].values()), wrench6, bool(f["suction_cmd"]),
                     bool(f.get("vacuum_sealed")))
        rec_pos = np.array(f["ee"]["pos"])
        rec_quat = np.array(f["ee"]["quat_wxyz"])
        pos_err = np.abs(s[:3] - rec_pos).max()
        qe = min(np.abs(s[3:7] - rec_quat).max(), np.abs(s[3:7] + rec_quat).max())
        moved = i > 0 and np.linalg.norm(rec_pos - np.array(frames[i - 1]["ee"]["pos"])) > 1e-4
        if moved:
            moving_pos = max(moving_pos, pos_err)
        else:
            static_pos = max(static_pos, pos_err)
            static_quat = max(static_quat, qe)
    print(f"  static frames: max |pos err| = {static_pos*1e6:.2f} um, "
          f"max |quat err| = {static_quat:.2e}")
    print(f"  moving frames: max |pos err| = {moving_pos*1e6:.2f} um "
          f"(recorder joint/EE read skew, not an FK error)")
    # "Static" allows up to 0.1 mm inter-frame motion, and the recorder's
    # joint-vs-EE read skew over that motion is tens of um — negligible vs the
    # mm-scale per-step actions. A real FK bug (wrong frame/joint order) shows
    # cm-scale errors, so 0.1 mm is a safe pass threshold; a true 0 needs the
    # genuinely-frozen hover frames that only some tasks have.
    ok = static_pos < 1e-4
    print(f"  -> {'OK, FK path reproduces recordings (within recorder read-skew)' if ok else 'MISMATCH — investigate before robot'}\n")

    # 2. integration round-trip: integrating the recorded action onto frame t's
    #    pose should land on frame t+1's recorded pose (validates Q2 convention).
    print("== delta-integration convention (recorded action: pose[t] -> pose[t+1]) ==")
    pos_resid = quat_resid = 0.0
    for i in range(len(frames) - 1):
        cur_pos = np.array(frames[i]["ee"]["pos"])
        cur_quat = np.array(frames[i]["ee"]["quat_wxyz"])
        nxt_pos = np.array(frames[i + 1]["ee"]["pos"])
        nxt_quat = np.array(frames[i + 1]["ee"]["quat_wxyz"])
        # recorded action = next-state delta, same formula as the converter
        dpos = nxt_pos - cur_pos
        w0, x0, y0, z0 = cur_quat
        w1, x1, y1, z1 = nxt_quat
        q_cur = Rotation.from_quat([x0, y0, z0, w0])
        q_nxt = Rotation.from_quat([x1, y1, z1, w1])
        drot = (q_nxt * q_cur.inv()).as_rotvec()
        act = np.concatenate([dpos, drot, [0.0]])
        pos_tgt, _, R_tgt = integrate(cur_pos, cur_quat, act)
        pos_resid = max(pos_resid, np.abs(pos_tgt - nxt_pos).max())
        q_err = (Rotation.from_matrix(R_tgt) * q_nxt.inv()).magnitude()
        quat_resid = max(quat_resid, q_err)
    print(f"  max pos residual  = {pos_resid*1e6:.2f} um")
    print(f"  max rot residual  = {quat_resid:.2e} rad")
    print(f"  -> {'OK, integration matches converter' if pos_resid < 1e-9 and quat_resid < 1e-6 else 'CONVENTION MISMATCH — fix integrate() before robot'}\n")

    # 3. policy predictions vs recorded actions on this take.
    print("== policy predicted vs recorded action (first 5 + summary) ==")
    policy, pre, post = load_policy(checkpoint)
    policy.reset()
    errs = []
    n = min(len(frames) - 1, len(rgb_paths))
    import cv2
    for i in range(n):
        img = cv2.cvtColor(cv2.imread(str(rgb_paths[i])), cv2.COLOR_BGR2RGB)
        f = frames[i]
        j = f["joints"]
        w = f["wrench"]
        s = ob.state(list(j["torso"].values()), list(j["left_arm"].values()),
                     list(j["right_arm"].values()),
                     [w["fx"], w["fy"], w["fz"], w["tx"], w["ty"], w["tz"]],
                     bool(f["suction_cmd"]), bool(f.get("vacuum_sealed")))
        depth_img = (ObsBuilder.depth_image(cv2.imread(str(depth_paths[i]), cv2.IMREAD_UNCHANGED))
                     if i < len(depth_paths) else None)
        pred = predict(policy, pre, post, s, ObsBuilder.image(img), instruction, depth_img)
        # recorded action
        cur_pos, nxt_pos = np.array(f["ee"]["pos"]), np.array(frames[i + 1]["ee"]["pos"])
        rec = nxt_pos - cur_pos
        if i < 5:
            print(f"  t={i:3d} pred dpos(mm)={pred[:3]*1000} suction={pred[6]:.2f} "
                  f"| rec dpos(mm)={rec*1000}")
        errs.append(np.abs(pred[:3] - rec))
    errs = np.array(errs)
    print(f"  mean |dpos err| over take = {errs.mean()*1000:.2f} mm "
          f"(matches eval_offline scale)\n")


# ── dry-run (live, no commands) ───────────────────────────────────────


def _goto_start(mover, take_dir: Path):
    """Move the arm to a recorded case_pick start pose via the demo's IK
    primitive (NOT the policy). The only motion this file performs, and only
    when --goto-start is given. Brings the EE into the trained workspace so the
    policy sees an in-distribution observation (recorded case_pick lives at
    x~0.69-0.77; a parked/home arm is far outside it)."""
    f0 = json.loads((take_dir / "states.jsonl").open().readline())
    pos = np.array(f0["ee"]["pos"])
    w, x, y, z = f0["ee"]["quat_wxyz"]
    rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz")
    print(f"--goto-start: moving arm to {take_dir.name} start pos={pos.round(3)} "
          f"via Mover.goto (not the policy)")
    mover.goto(pos, rpy, step_duration=3.0)
    print("  arm at start pose.")


def _force_mag(mover) -> float:
    """Live raw wrench force magnitude (N). Carries a ~14 N gravity offset."""
    ws = getattr(mover._arm, "wrench_sensor", None)
    if ws is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(ws.get_state()["wrench"], float)[:3]))


def _baseline_force(mover, n: int = 5) -> float:
    """Mean raw force over n reads — the payload-aware reference for the force
    guard. Re-taken at EACH task start so carrying a case/battery doesn't bias
    the contact reading (the demo re-tares the same way after a pick)."""
    return float(np.mean([_force_mag(mover) for _ in range(n)]))


def _retreat_to_hover(mover, cap_z: float | None = None, by: float = 0.12):
    """Back the EE straight up by `by` metres (x, y and orientation held) to
    relieve a downward contact after an abort/stall — a BOUNDED relative lift,
    NOT a move to an absolute hover.

    Aborts often leave the arm at a workspace-boundary / near-reach-limit pose
    (the closed-loop run drifted OOD before it stopped). From there a strict
    hold-x,y climb to an absolute high z is kinematically infeasible: the solver
    caves the lateral hold and the arm slews off-target (observed: a 0.84->1.10
    'lift' ended ~0.6 m off as y collapsed from the 0.60 box edge). A small
    relative back-off stays reachable and just lifts the cup off contact; the
    next run's go_to_default_pose does the full clearance/reset.

    cap_z (the highest reference target z commanded this run, i.e. the hover the
    arm descended from) bounds the lift: the cup backs UP toward it but never
    above it, so an abort already high up can't be pushed past the workspace.

    Uses the demo's vertical-Z primitive (mover.lift): holds x, y and
    orientation and pins the IK posture to the start config so the solve stays
    on one branch (no goto/mid-range-posture branch jump). No-op if the capped
    target is at or below the current z."""
    pos, _ = mover.current_ee_pose()
    target_z = pos[2] + by
    if cap_z is not None:
        target_z = min(target_z, float(cap_z))
    print(f"  retreating: z {pos[2]:.2f} -> {target_z:.2f} (x, y held"
          f"{f', capped at prev target {float(cap_z):.2f}' if cap_z is not None else ''})")
    mover.lift(z=target_z)


class RolloutLog:
    """Persist a live rollout to the recorder's states.jsonl / meta.json layout
    so Research/gradual_drift can profile its deviation. One take dir per
    sub-task (mirroring the demo recordings), named <stamp>_ep<NN>_<task>."""

    def __init__(self, root: Path, checkpoint: Path, save_images: bool = False,
                 run_num: int = 0):
        self.root = Path(root)
        suffix = f"_r{run_num:02d}" if run_num > 1 else ""
        self.stamp = time.strftime("%Y%m%d-%H%M%S") + suffix
        self.checkpoint = str(checkpoint)
        self.save_images = save_images
        self.f = None
        self.take_dir = None
        self.task = self.instruction = None
        self.n = 0

    def open_task(self, ep: int, task: str, instruction: str):
        self.close()
        self.take_dir = self.root / f"{self.stamp}_ep{ep:04d}_{task}"
        self.take_dir.mkdir(parents=True, exist_ok=True)
        self.f = (self.take_dir / "states.jsonl").open("w")
        self.task, self.instruction, self.n = task, instruction, 0
        if self.save_images:
            (self.take_dir / "head_rgb").mkdir(exist_ok=True)
            (self.take_dir / "head_depth").mkdir(exist_ok=True)

    def frame(self, t, state, wrench6, pred, clamped, chunk_boundary,
              rgb=None, depth_m=None):
        if self.f is None:
            return
        if self.save_images and rgb is not None:
            # Match the recorder byte-for-byte so load_take/_load_rgb/_load_depth
            # replay it like a demo: BGR jpg(q95) + uint16-mm PNG (0 = invalid).
            import cv2
            stem = f"{self.n:06d}"
            cv2.imwrite(str(self.take_dir / "head_rgb" / f"{stem}.jpg"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if depth_m is not None:
                d = np.asarray(depth_m, dtype=np.float32)
                if d.ndim == 3:
                    d = d[..., 0]
                mm = np.clip(np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0, neginf=0.0),
                             0, 65535).astype(np.uint16)
                cv2.imwrite(str(self.take_dir / "head_depth" / f"{stem}.png"), mm)
        self.f.write(json.dumps({
            "i": self.n, "t": float(t),
            "ee": {"pos": [float(x) for x in state[:3]],
                   "quat_wxyz": [float(x) for x in state[3:7]]},
            "wrench": {k: float(v) for k, v in
                       zip(("fx", "fy", "fz", "tx", "ty", "tz"), wrench6)},
            "suction_cmd": bool(state[7] > 0.5),
            "vacuum_sealed": bool(state[8] > 0.5),
            "chunk_boundary": bool(chunk_boundary),
            "action_pred": [float(x) for x in pred],
            "action_cmd": [float(x) for x in clamped],
        }) + "\n")
        self.n += 1

    def close(self, success: bool | None = None):
        if self.f is None:
            return
        self.f.close()
        self.f = None
        meta = {
            "phase": self.task, "instruction": self.instruction,
            "action_space": "ee_delta+suction", "ee_rotation": "quat_wxyz, base_link",
            "source": "run_policy.py rollout", "checkpoint": self.checkpoint,
            "frames": self.n, "success": success,
        }
        if self.save_images:
            meta["depth_units"] = "uint16 millimetres (0 = invalid)"
        (self.take_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _load_mpc(ckpt: Path, train_root: Path, horizon: int, samples: int, iters: int):
    """Load the MPC latent-dynamics planner (MPC/planner.py) + the per-phase
    data-derived contact targets. The MPC modules are flat (no package), so put
    MPC/ on sys.path; their names don't clash with anything already imported."""
    sys.path.insert(0, str(LGES_DIR.parent / "MPC"))
    from data import Normalizer
    from model import LatentDynamics
    from reward import compute_targets, PhaseReward
    from planner import MPPIPlanner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = blob["config"]
    model = LatentDynamics(cfg["latent"], cfg["hidden"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    norm = Normalizer.from_dict(blob["norm"])
    targets = compute_targets(train_root)
    # reward is reassigned per sub-task in the loop; seed with any phase.
    planner = MPPIPlanner(model, norm, PhaseReward(next(iter(targets.values()))),
                          device, horizon, samples, iters)
    for _ in range(3):  # warm up CUDA kernels so the first live tick isn't ~300 ms
        planner.plan(np.zeros(15, np.float32))
    planner.reset()
    print(f"MPC planner: {ckpt}\n  H={horizon} N={samples} iters={iters} | device {device}")
    return planner, targets, PhaseReward


def run_live(checkpoint: Path, tasks: list[str], *, commit: bool,
             goto_start: Path | None = None, home: bool = True,
             force_limit_n: float | None = None, max_ticks: int = 350,
             pause_between: bool = False, enforce_box: bool = False,
             log_dir: Path | None = None, mpc: bool = False,
             mpc_train: Path | None = None, mpc_horizon: int = 15,
             mpc_samples: int = 256, mpc_iters: int = 3,
             log_images: bool = False, n_action_steps: int | None = None,
             loop: bool = False, descend_until_contact: bool = False,
             contact_n: float = 3.0, descend_floor: float = 0.76,
             descend_rate: float = 0.006, film: bool = False):
    """Run one or more suction sub-tasks in sequence at ~15 Hz.

    commit=False -> DRY-RUN (prints, commands nothing).
    commit=True  -> COMMANDS the left arm + suction, fully guarded.

    Action source: the SmolVLA policy, or (mpc=True) the MPPI planner over the
    latent dynamics model. MPC needs only the 15-dim state, so cameras and the
    policy are skipped and the reference re-grounds to the live pose every tick.

    Per task: switch instruction + workspace box, reset the action source,
    re-baseline the force guard, then loop until the task's done-signal (pick =
    vacuum seal; place = release after holding), a safety abort, or the per-task
    tick cap. Tasks flow continuously — each ends at the hover where the next begins.
    """
    mode = "GO — COMMANDS THE LEFT ARM" if commit else "DRY-RUN — policy commands nothing"
    print(f"{mode}. sequence: {' -> '.join(tasks)}\ncheckpoint: {checkpoint}")
    from dexcontrol.robot import Robot
    from dexcontrol.core.config import get_robot_config
    # case_battery_demo modules use package-relative imports -> import as package.
    from case_battery_demo.grasp import SuctionMover  # noqa: E402  IK + live reads
    from case_battery_demo.home_pose import go_to_default_pose  # noqa: E402
    from case_battery_demo import suction_io, config as cfg  # noqa: E402

    ob = ObsBuilder()
    if mpc:
        planner, mpc_targets, PhaseReward = _load_mpc(
            checkpoint, mpc_train, mpc_horizon, mpc_samples, mpc_iters)
        policy = pre = post = None
        chunk_steps = 1  # MPC replans every tick -> reground the reference each tick
    else:
        planner = mpc_targets = PhaseReward = None
        policy, pre, post = load_policy(checkpoint, film=film)
        if n_action_steps is not None:
            policy.config.n_action_steps = n_action_steps
        chunk_steps = int(getattr(policy.config, "n_action_steps", 1))

    # The head camera is disabled in the default config; enable it (over zenoh)
    # so we get frames, exactly as run_demo.py does for the dashboard/recorder.
    # MPC uses only the 15-dim state, so it needs no camera.
    robot_configs = get_robot_config()
    if not mpc:
        robot_configs.enable_sensor("head_camera")
        robot_configs.sensors["head_camera"].transport = "zenoh"
    flim = cfg.FORCE_HARD_LIMIT_N if force_limit_n is None else force_limit_n
    with Robot(configs=robot_configs) as bot, SuctionMover(bot) as mover:
        # ONE-TIME SESSION SETUP: head tilt, auth prompt, vacuum monitor.
        # Load LGES/utils.py by path: a bare `import utils` resolves to
        # grasp_box/utils.py (grasp.py puts grasp_box on sys.path for read_force).
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "lges_utils", LGES_DIR / "utils.py")
        _lges_utils = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_lges_utils)
        _lges_utils.set_head_pitch(bot, pitch_deg=30.0)  # tilt head to see workspace

        if commit:
            print(f"\n  *** GO COMMANDS THE LEFT ARM. Keep a hand on the e-stop. ***\n"
                  f"  clamps dpos<={(MAX_DPOS_M*1000).round(0)} mm / "
                  f"drot<={MAX_DROT_RAD*1000:.0f} mrad | force abort +{flim:.0f}N | "
                  f"workspace box {'ON' if enforce_box else 'OFF'} | "
                  f"<= {max_ticks} ticks/task | {'PAUSE between tasks' if pause_between else 'continuous'}"
                  + (" | LOOP" if loop else "")
                  + (f" | DESCEND-UNTIL-CONTACT gate (stop@{contact_n:.0f}N, floor {descend_floor:.2f}m)"
                     if descend_until_contact else ""))
            try:
                input("  >>> ENTER to authorize motion (Ctrl-C to cancel) <<< ")
            except (KeyboardInterrupt, EOFError):
                print("cancelled — no motion."); return

        # Vacuum-seal monitor: shared across loop runs (one start/stop per session).
        seal = None
        try:
            seal = suction_io.VacuumMonitor(); seal.start()
        except Exception as e:  # noqa: BLE001
            print(f"  (vacuum monitor unavailable: {e}; seal -> 0, tasks stop on max-ticks)")

        dt = 1.0 / FPS
        run_num = 0
        try:
          while True:
            run_num += 1
            if run_num > 1:
                print(f"\n{'='*20} run {run_num} {'='*20}")

            # Home both arms then goto-start every run.
            if home:
                go_to_default_pose(bot)
            if goto_start is not None:
                _goto_start(mover, goto_start)

            # Per-run: fresh log timestamp.
            log = RolloutLog(log_dir, checkpoint, save_images=log_images, run_num=run_num) if log_dir is not None else None
            # When pausing at each boundary, the per-task ENTER prompt is the
            # control point; otherwise poll stdin each tick (see the tick loop):
            # 'q' breaks the current run, any other line is the --go ENTER-abort.
            watch_stdin = not pause_between and (commit or loop)

            # Highest commanded reference-target z this run — the hover the arm
            # descended from. An abort retreat backs UP toward it but never
            # above it (see _retreat_to_hover).
            max_tgt_z = -np.inf
            run_stop = None
            try:
                for ti, task in enumerate(tasks):
                    spec = TASKS[task]
                    instruction, kind, box = spec["instruction"], spec["kind"], spec["box"]
                    if mpc:
                        planner.reward = PhaseReward(mpc_targets[task]).to(planner.device)
                        planner.reset()  # clear the warm-started plan between sub-tasks
                    else:
                        policy.reset()  # clear the action-chunk queue between sub-tasks
                    baseline_f = _baseline_force(mover)
                    entry_sealed = bool(seal.is_sealed()) if seal else False
    
                    if commit and pause_between:
                        try:
                            input(f"\n  >>> [{ti+1}/{len(tasks)}] ENTER to run '{task}' "
                                  f"(Ctrl-C to stop) <<< ")
                        except (KeyboardInterrupt, EOFError):
                            run_stop = "operator stopped at task boundary"; break
                    print(f"\n=== task {ti+1}/{len(tasks)}: {task} ({kind}) | \"{instruction}\"\n"
                          f"    baseline {baseline_f:.1f}N | box {box[0]}..{box[1]}"
                          f"{' | entry sealed' if entry_sealed else ''} ===")
    
                    if log is not None:
                        log.open_task(ti, task, instruction)
                    task_done = None
                    has_sealed = entry_sealed   # pick: latch the grasp
                    has_released = False        # place: latch the release
                    went_low = False            # latch that the arm descended to the object
                    ref_pos = ref_quat = None   # running reference target (see below)
                    for tick in range(max_ticks):
                        t0 = time.time()
                        torso_q = bot.torso.get_joint_pos()
                        left_q = bot.left_arm.get_joint_pos()
                        right_q = bot.right_arm.get_joint_pos()
                        ws = getattr(mover._arm, "wrench_sensor", None)
                        wrench6 = (np.asarray(ws.get_state()["wrench"], float)[:6]
                                   if ws is not None else np.zeros(6))
                        t_read = time.time()
                        if mpc:
                            rgb = depth_m = None      # MPC needs no camera
                        else:
                            rgb = _grab_rgb(bot)
                            depth_m = _grab_depth(bot)
                        suction = suction_io.is_suction_commanded_on()
                        sealed = bool(seal.is_sealed()) if seal else False
                        t_cam = time.time()

                        state = ob.state(torso_q, left_q, right_q, wrench6, suction, sealed)
                        if mpc:
                            pred = planner.plan(state)
                        else:
                            pred = predict(policy, pre, post, state, ObsBuilder.image(rgb),
                                           instruction, ObsBuilder.depth_image(depth_m))
                        t_inf = time.time()
                        clamped = clamp_action(pred)
                        clip = "CLIP" if not np.allclose(clamped, pred) else "    "
                        cur_pos, cur_quat = state[:3], state[3:7]
                        contact = float(np.linalg.norm(wrench6[:3])) - baseline_f
                        has_sealed = has_sealed or sealed
                        # Privileged descend-until-contact gate (diagnostic): while a pick
                        # is descending and not yet in contact/sealed, ensure the EE keeps
                        # going DOWN rather than stopping at the policy's habitual depth.
                        # Overrides ONLY the z-delta (lateral/rotation/suction stay from the
                        # policy, so alignment failures aren't masked). Stops the instant
                        # contact rises or seal forms; hard-guarded by descend_floor and the
                        # force-limit abort below.
                        gating = (descend_until_contact and kind == "pick"
                                  and not has_sealed and contact < contact_n
                                  and descend_floor < cur_pos[2] < DESCEND_Z)
                        if gating:
                            clamped[2] = min(clamped[2], -descend_rate)
                        # Integrate deltas onto a running REFERENCE target, NOT the
                        # live pose: at 15 Hz the arm lags the target, so cur_pos+dpos
                        # under-advances and the motion stalls. The policy emits a
                        # 50-step open-loop chunk and only re-reads the observation
                        # at chunk boundaries (every chunk_steps ticks), so re-ground
                        # the reference to the live pose there (a small, safe backward
                        # correction) and advance it purely by the deltas in between.
                        if tick % chunk_steps == 0:
                            ref_pos, ref_quat = cur_pos.copy(), cur_quat.copy()
                        pos_tgt, rpy_tgt, R_tgt = integrate(ref_pos, ref_quat, clamped)
                        ref_pos, ref_quat = pos_tgt, _rpy_to_quat_wxyz(rpy_tgt)
                        max_tgt_z = max(max_tgt_z, float(pos_tgt[2]))
                        if log is not None:
                            log.frame(t0, state, wrench6, pred, clamped, tick % chunk_steps == 0,
                                      rgb=rgb, depth_m=depth_m)
                        in_box = workspace_ok(pos_tgt, box)
                        q, ok = mover._solve_ik(mover._fresh_configuration(), pos_tgt, rpy_tgt)
                        t_ik = time.time()
                        went_low = went_low or cur_pos[2] < DESCEND_Z
                        if entry_sealed and not sealed:
                            has_released = True
                        at_hover = cur_pos[2] >= HOVER_Z
    
                        # Diagnostic: c-hat the FiLM actually saw for this chunk (set in
                        # embed_prefix at the chunk boundary). Confirms whether contact/seal
                        # rises at the hard push (-> if dpos stays negative, FiLM has no
                        # authority = architecture) or never rises (-> obs-pipeline bug).
                        chat_str = ""
                        if film:
                            cc = getattr(policy.model, "_cur_contact", None)
                            if cc is not None:
                                chat_str = " c^=[" + ",".join(f"{v:.2f}" for v in cc.flatten().tolist()) + "]"
                        print(f"[{ti+1}.{tick:3d}] dpos(mm)={pred[:3]*1000} suc={pred[6]:.2f} {clip} "
                              f"{'GATE' if gating else '    '} | "
                              f"tgt={pos_tgt.round(3)} z={cur_pos[2]:.2f} "
                              f"box={('ok' if in_box else 'OUT') if enforce_box else 'off'} "
                              f"ik={'ok' if ok else 'best'} contact={contact:+.1f}N "
                              f"seal={'Y' if sealed else '.'}{chat_str} | "
                              f"rd={(t_read-t0)*1000:.0f} cam={(t_cam-t_read)*1000:.0f} "
                              f"inf={(t_inf-t_cam)*1000:.0f} ik={(t_ik-t_inf)*1000:.0f}ms")
    
                        # Task-done (success) — checked before aborts. Completes only
                        # after the grasp/release AND the lift/retract back to hover,
                        # so the case is actually lifted and the next task starts
                        # in-distribution. Suction is NOT dropped here (a pick leaves
                        # the object held for the next task).
                        if kind == "pick" and has_sealed and went_low and at_hover:
                            task_done = f"grasped+lifted@{tick}"; break
                        if kind == "place" and has_released and went_low and at_hover:
                            task_done = f"placed+retracted@{tick}"; break

                        # Operator stdin (non-blocking, after task-done so a
                        # finishing task isn't pre-empted): a line of just 'q'
                        # breaks this run — under --loop that drops back to the
                        # rerun prompt rather than aborting. Any other line is
                        # the --go ENTER-abort. EOF (empty read) is ignored.
                        if watch_stdin and select.select([sys.stdin], [], [], 0)[0]:
                            line = sys.stdin.readline()
                            if line.strip().lower() == "q":
                                run_stop = "operator (q)"; break
                            if line and commit:
                                run_stop = "ABORT: operator (ENTER)"; break

                        if commit:
                            if contact > flim:
                                run_stop = f"ABORT: force {contact:.1f}N > +{flim:.1f}N"; break
                            if enforce_box and not in_box:
                                run_stop = f"ABORT: target {pos_tgt.round(3)} out of '{task}' box"; break
                            # Command the best-effort IK solution (like the demo);
                            # non-convergence is fine. Abort only on a genuine blow-up:
                            # a large one-tick joint jump (near-singular).
                            arm_q_cmd = mover._arm_joints_from_q(q)
                            dq = float(np.abs(arm_q_cmd - left_q).max())
                            if dq > MAX_JOINT_STEP_RAD:
                                run_stop = f"ABORT: joint jump {dq:.2f} rad > {MAX_JOINT_STEP_RAD} (IK near-singular?)"; break
                            mover._arm.set_joint_pos(arm_q_cmd)
                            # Suction commands hit a weblogic endpoint that blocks
                            # ~0.5 s per call (suction_io._run), so command it only on
                            # a state CHANGE, not every tick — otherwise the loop is
                            # pinned at ~2 Hz by the suction call alone.
                            want_suction = pred[6] > 0.5
                            if want_suction != suction_io.is_suction_commanded_on():
                                (suction_io.suction_on if want_suction else suction_io.suction_off)()
    
                        work = time.time() - t0
                        if work > dt:
                            print(f"      SLOW tick {tick}: full work={work*1000:.0f}ms "
                                  f"(budget {dt*1000:.0f}ms) — tail after ik={(work-(t_ik-t0))*1000:.0f}ms")
                        time.sleep(max(0.0, dt - work))
    
                    if log is not None:
                        log.close(success=task_done is not None)
                    if run_stop:
                        break
                    if task_done is None:
                        run_stop = f"STALL: '{task}' did not finish in {max_ticks} ticks"; break
                    print(f"  -> '{task}' done ({task_done})")
                else:
                    run_stop = "sequence complete"
            except KeyboardInterrupt:
                run_stop = "ABORT: KeyboardInterrupt"
            finally:
                if log is not None:
                    log.close(success=False)

            if commit:
                if run_stop == "sequence complete":
                    # Clean finish: arm is at hover holding the object. Drop suction
                    # so the case releases before the next loop run starts.
                    suction_io.suction_off()
                else:
                    # Abort/stall: drop suction first then retreat to hover.
                    suction_io.suction_off()
                    try:
                        cap = max_tgt_z if np.isfinite(max_tgt_z) else None
                        _retreat_to_hover(mover, cap_z=cap)
                    except KeyboardInterrupt:
                        print("  retreat interrupted by operator")
                    except Exception as e:  # noqa: BLE001
                        print(f"  retreat failed ({e}); leaving arm in place")

            if not commit:
                tail = "no motion commanded; arm left where it stopped."
            elif run_stop == "sequence complete":
                tail = "arm at hover, suction off."
            else:
                tail = "retreated to hover; suction off."
            print(f"\n{'GO' if commit else 'DRY-RUN'} ended: {run_stop}. {tail}")

            if not loop or run_stop == "ABORT: KeyboardInterrupt":
                break
            try:
                input("\n>>> Press Enter to run again (Ctrl-C to stop) <<< ")
            except (KeyboardInterrupt, EOFError):
                print("\nloop stopped."); break

        finally:
            if seal is not None:
                seal.stop()


def _grab_rgb(bot):
    """Grab the head camera's left RGB frame — same source the recorder used."""
    if not bot.has_sensor("head_camera"):
        raise SystemExit("head_camera not available — enable it in the robot config")
    rgb = bot.sensors.head_camera.get_left_rgb()
    if rgb is None:
        raise SystemExit("head_camera returned no frame (is the dexsensor running?)")
    return rgb


def _grab_depth(bot):
    """Grab the head camera's depth frame (metres) — same source the recorder
    used; ObsBuilder.depth_image converts to the colorized camera2."""
    depth = bot.sensors.head_camera.get_depth()
    if depth is None:
        raise SystemExit("head_camera returned no depth (is the dexsensor running?)")
    return depth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--self-test", type=Path, metavar="TAKE_DIR",
                    help="offline validation against a recorded take (no robot)")
    ap.add_argument("--dry-run", action="store_true",
                    help="live print-only loop (needs robot, policy commands nothing)")
    ap.add_argument("--go", action="store_true",
                    help="live loop that COMMANDS the left arm + suction (guarded). "
                         "Use --goto-start to begin in-distribution; otherwise the "
                         "arm starts where it is and an out-of-box target aborts.")
    ap.add_argument("--task", default=None, choices=list(TASKS),
                    help="run a single sub-task (default: case_pick). Note: a "
                         "standalone 'place' has nothing held and will stall.")
    ap.add_argument("--chain", nargs="+", metavar="TASK", default=None,
                    help=f"run these sub-tasks in sequence, e.g. "
                         f"--chain case_pick case_place. Use 'all' for the full "
                         f"forward sequence. Choices: {list(TASKS)}")
    ap.add_argument("--goto-start", type=Path, metavar="TAKE_DIR", default=None,
                    help="after homing, move the arm to this take's start pose "
                         "via Mover.goto (the only EE-space motion before the loop)")
    ap.add_argument("--no-home", action="store_true",
                    help="skip the go_to_default_pose homing step at startup")
    ap.add_argument("--pause-between", action="store_true",
                    help="(--go chains) wait for ENTER before each sub-task so you "
                         "can verify each handoff; disables the mid-run ENTER-abort")
    ap.add_argument("--box", action="store_true",
                    help="enforce the per-task workspace box (abort if a target "
                         "leaves it). Off by default; force limit + clamps still apply.")
    ap.add_argument("--force-limit", type=float, default=None,
                    help="abort if contact force exceeds baseline by this many N "
                         "(default cfg.FORCE_HARD_LIMIT_N=20; use ~8 for first runs)")
    ap.add_argument("--max-ticks", type=int, default=1000,
                    help="per-task tick cap (episodes are ~230-275 frames)")
    ap.add_argument("--log-dir", type=Path, default=None, metavar="DIR",
                    help="persist the rollout to DIR as states.jsonl/meta.json "
                         "takes (one per sub-task) for Research/gradual_drift")
    ap.add_argument("--log-images", action="store_true",
                    help="with --log-dir, also save per-frame head_rgb/ + head_depth/ "
                         "(recorder format) so the rollout is replayable by the policy")
    ap.add_argument("--mpc", action="store_true",
                    help="use the MPC latent-dynamics planner (MPC/planner.py) "
                         "instead of the SmolVLA policy. Needs only the 15-dim "
                         "state, so cameras and the policy are skipped.")
    ap.add_argument("--mpc-ckpt", type=Path, default=LGES_DIR.parent / "MPC/runs/dyn/best.pt",
                    help="dynamics-model checkpoint for --mpc")
    ap.add_argument("--mpc-train", type=Path, default=VLA_DIR / "datasets/lges_suction",
                    help="training dataset for --mpc data-derived contact targets")
    ap.add_argument("--mpc-horizon", type=int, default=15, help="MPPI planning horizon")
    ap.add_argument("--mpc-samples", type=int, default=256, help="MPPI samples/iter")
    ap.add_argument("--mpc-iters", type=int, default=3, help="MPPI refinement iters")
    ap.add_argument("--loop", action="store_true",
                    help="after each run completes (or aborts), prompt Enter to run again; "
                         "type 'q'+Enter mid-run to break it and return to that prompt; "
                         "Ctrl-C exits. Useful for repeated rollouts without restarting.")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override policy n_action_steps (chunk open-loop length). "
                         "Lower values (e.g. 5) approach closed-loop; default uses the "
                         "trained value (50). Useful for isolating chunk-latency effects.")
    ap.add_argument("--film", action="store_true",
                    help="load a FiLM contact-conditioned checkpoint (V1/V2 from train_film.sh): "
                         "patches the model so c-hat is computed live + the wrench is masked from "
                         "the action path. Use with --checkpoint outputs/film_v2/checkpoints/last.")
    ap.add_argument("--descend-until-contact", action="store_true",
                    help="diagnostic gate: during a pick, override the z-delta to keep the EE "
                         "descending until contact/seal (tests whether forcing condition-use "
                         "recovers under-reach). Lateral/rotation/suction stay from the policy.")
    ap.add_argument("--contact-n", type=float, default=3.0,
                    help="contact-gate stops forcing descent when contact force (baseline-"
                         "subtracted N) exceeds this (default 3.0)")
    ap.add_argument("--descend-floor", type=float, default=0.76,
                    help="contact-gate hard z-floor (m): never force descent below this "
                         "(default 0.76, just under the deepest demo 0.776)")
    ap.add_argument("--descend-rate", type=float, default=0.0003,
                    help="min downward z step (m/tick) the contact-gate enforces (default 0.006)")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.self_test, args.checkpoint or latest_checkpoint())
        return
    if not (args.dry_run or args.go):
        ap.error("choose --self-test TAKE_DIR, --dry-run, or --go")
    if args.chain and args.task:
        ap.error("use either --task or --chain, not both")
    if args.chain == ["all"]:
        args.chain = list(SUCTION_SEQUENCE)  # full case+battery forward sequence
    tasks = args.chain or [args.task or "case_pick"]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        ap.error(f"unknown task(s) {bad}; choices: {list(TASKS)} (or 'all')")
    if args.go and args.goto_start is None:
        print("[warn] --go without --goto-start: the arm starts wherever it is. "
              "Position it in-distribution first" +
              (" (with --box, an out-of-box target aborts immediately)." if args.box else "."))
    # MPC uses its own dynamics-model checkpoint, not a SmolVLA one.
    ckpt = args.mpc_ckpt if args.mpc else (args.checkpoint or latest_checkpoint())

    run_live(ckpt, tasks, commit=args.go, goto_start=args.goto_start,
             home=not args.no_home, force_limit_n=args.force_limit,
             max_ticks=args.max_ticks, pause_between=args.pause_between,
             enforce_box=args.box, log_dir=args.log_dir, log_images=args.log_images,
             mpc=args.mpc, mpc_train=args.mpc_train, mpc_horizon=args.mpc_horizon,
             mpc_samples=args.mpc_samples, mpc_iters=args.mpc_iters,
             n_action_steps=args.n_action_steps, loop=args.loop,
             descend_until_contact=args.descend_until_contact, contact_n=args.contact_n,
             descend_floor=args.descend_floor, descend_rate=args.descend_rate,
             film=args.film)


if __name__ == "__main__":
    main()


#FILM_COND=contact FILM_MASK_FORCE=0 FILM_INJECT=suffix /home/dexmate/vla_venv/bin/python run_policy.py --go --task case_pick --goto-start /home/dexmate/CNS_code/Dexmate/LGES/recordings/case_pick/20260617-145810_ep0001_case_pick --checkpoint outputs/film_v2_contact_suffix/checkpoints/last --film