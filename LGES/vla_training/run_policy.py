#!/usr/bin/env python3
"""On-robot executor for the trained SmolVLA policy — ik_demo + detection stack.

Rebuilt (2026-07) on LGES/ik_demo (hardened pink IK, drivers.suction_io) and the
BEV case detection, replacing the case_battery_demo plumbing. Matches the data
collected by collect_case_pick.py: episodes start at the view-park pose with the
case detected; the workspace box derives from the detection instead of taught
poses, and --goto-start is gone (home -> view park IS the in-distribution start).

Scope: case_pick, left arm, suction. Three modes:

  --self-test TAKE_DIR   offline, no robot. Rebuilds the 15-dim observation
                         from a recorded take's joints (the same path the
                         live loop uses) and checks it reproduces the take's
                         stored state, then runs the policy and prints
                         predicted-vs-recorded actions.

  --dry-run              live, needs the robot. Loops at ~15 Hz: reads live
                         observation, runs the policy, prints predicted action
                         + safety-clamped target + IK feasibility. Commands
                         NOTHING.

  --go                   live. Same loop but COMMANDS the left arm (IK ->
                         set_joint_pos) and suction, fully guarded (per-step
                         clamp, workspace box, force limit, joint-jump abort,
                         operator abort) and stopping on vacuum seal.

Model-side API (load_policy, predict, _to_chw, ObsBuilder, integrate,
clamp_action, RolloutLog) is unchanged — Research/ probes import these.
The old executor (case_battery_demo Mover, --goto-start, MPC mode, task
chains) lives in git history if the old scene ever needs re-running.

Run with the vla_venv python (imports lerobot AND the ik/detection stack):
  /home/dexmate/vla_venv/bin/python LGES/vla_training/run_policy.py --self-test <take_dir>
  /home/dexmate/vla_venv/bin/python LGES/vla_training/run_policy.py --dry-run
  /home/dexmate/vla_venv/bin/python LGES/vla_training/run_policy.py --go --force-limit 8
"""

import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

VLA_DIR = Path(__file__).resolve().parent
LGES_DIR = VLA_DIR.parent
REPO_DIR = LGES_DIR.parent
sys.path.insert(0, str(REPO_DIR))  # LGES.ik_demo / detection (via chassis_sequence)
sys.path.insert(0, str(VLA_DIR))   # convert_to_lerobot, collect_case_pick, film_contact

# Reuse the converter's exact depth colorize so live obs == training data.
from convert_to_lerobot import colorize_depth  # noqa: E402

IMG_W, IMG_H = 512, 320  # must match convert_to_lerobot.py
FPS = 10

# Per-step safety clamps. These bound a single step but must NOT throttle
# normal motion: at half these values the arm just hovered (real predicted
# deltas got clipped and never reached the case). [10,30,30] mm clears a full
# case_pick while staying within the training action range, so it still
# catches out-of-distribution spikes.
MAX_DPOS_M = np.array([0.01, 0.03, 0.03])
MAX_DROT_RAD = 0.025

# A pick isn't done at the seal instant — the recorded episodes end AFTER the
# lift back to SAFE_TRANSPORT_Z (1.12). So a task completes only once it has
# sealed AND the EE is back at hover height. DESCEND_Z guards a seal flicker
# at episode start from false-completing: the arm must actually have gone down
# to the object first. New-scene geometry: contact z spans ~0.74-0.82 (5-layer
# box stack), pre-descend hover = contact + HOVER_HEIGHT (0.25) >= 0.99, lift
# ends at 1.12 -> DESCEND_Z below the deepest hover, HOVER_Z below the lift top.
HOVER_Z = 1.05
DESCEND_Z = 0.95

# IK non-convergence is NOT a failure — solve_pose returns a best-effort joint
# config (the scripted ik_demo legs command it the same way). The real danger
# is a genuine blow-up near a singularity: a large one-tick joint jump.
MAX_JOINT_STEP_RAD = 1.2

# Per-task config: instruction the policy trained on + kind (pick ends on
# vacuum seal / place ends on release). The workspace box is DYNAMIC — derived
# from the BEV detection each run (see _box_from_detection) — because the case
# can be anywhere the chassis parked it, unlike the old taught-pose scene.
TASKS = {
    "case_pick": dict(instruction="pick up the case with the suction cup", kind="pick"),
}


# ── observation builder (shared by self-test and live loop) ──────────


def _rpy_to_quat_wxyz(rpy) -> np.ndarray:
    """Exactly the recorder's conversion (collect_case_pick._rpy_to_quat_wxyz)."""
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class ObsBuilder:
    """Reconstructs the state + head RGB + colorized head depth the policy
    expects (matches convert_to_lerobot.py exactly).

    Uses collect_case_pick._EEKin — the full-URDF FK that produces the training
    `states.jsonl` — so live observations match the recordings. State layout:
    pos(3) quat_wxyz(4) suction(1) vacuum_sealed(1) raw-wrench fx..tz(6).
    """

    def __init__(self, df_channel: bool = False):
        from collect_case_pick import _EEKin
        self._fk = _EEKin()
        # dfmag (d|F|/dt, N/frame) 16th state dim for *_dF checkpoints. Set
        # df_channel after load_policy: robot_state_feature.shape[0] == 16.
        self.df_channel = df_channel
        self._prev_fmag = None
        self._prev_t = 0.0

    def state(self, torso_q, left_q, right_q, wrench6, suction_on: bool,
              sealed: bool) -> np.ndarray:
        # right_q kept for signature stability; the suction arm's EE is the
        # only pose in the state vector.
        pos, rpy = self._fk.compute(torso_q, left_q)
        quat = _rpy_to_quat_wxyz(rpy)
        # Canonical sign, matching convert_to_lerobot.Q_REF: the grasp's
        # roll~pi puts qw~0, so the raw conversion flips to the antipode when
        # live roll wobbles across +/-180 deg — a state discontinuity training
        # never saw. Anchor on qx (= cos(yaw/2), always positive in this demo).
        if quat[1] < 0:
            quat = -quat
        s = np.concatenate([
            np.asarray(pos, dtype=np.float64),
            quat,
            [1.0 if suction_on else 0.0],
            [1.0 if sealed else 0.0],
            np.asarray(wrench6, dtype=np.float64)[:6],
        ])
        if self.df_channel:
            import time
            fmag = float(np.linalg.norm(np.asarray(wrench6, dtype=np.float64)[:3]))
            now = time.monotonic()
            # >0.5s since the last frame = a new rollout (loop runs at ~15Hz);
            # matches training where dfmag=0 on each episode's first frame.
            stale = self._prev_fmag is None or (now - self._prev_t) > 0.5
            dfmag = 0.0 if stale else fmag - self._prev_fmag
            self._prev_fmag, self._prev_t = fmag, now
            s = np.concatenate([s, [dfmag]])
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


def clamp_abs_action(pred: np.ndarray, cur_pos: np.ndarray, cur_quat_wxyz: np.ndarray):
    """Safety-clamp an absolute-action policy's prediction (pred = [x,y,z,
    qw,qx,qy,qz,suction], matching convert_to_lerobot.py's --action-space abs).
    Unlike delta actions, the model's raw target is an ABSOLUTE pose with no
    built-in per-tick bound, so cap the IMPLIED step from the live pose to the
    same MAX_DPOS_M/MAX_DROT_RAD clamp_action() uses. Returns (pos_tgt, rpy_tgt,
    dpos_clamped, drot_clamped, clipped) -- the deltas are for RolloutLog/print
    parity with the delta path, not because abs actions integrate onto a
    running reference (they don't: each chunk step is already a target)."""
    pred_pos = pred[0:3]
    pred_quat = pred[3:7] / np.linalg.norm(pred[3:7])
    raw_dpos = pred_pos - cur_pos
    dpos = np.clip(raw_dpos, -MAX_DPOS_M, MAX_DPOS_M)
    w0, x0, y0, z0 = cur_quat_wxyz
    w1, x1, y1, z1 = pred_quat
    q_cur = Rotation.from_quat([x0, y0, z0, w0])
    q_tgt = Rotation.from_quat([x1, y1, z1, w1])
    raw_drot = (q_tgt * q_cur.inv()).as_rotvec()
    drot = np.clip(raw_drot, -MAX_DROT_RAD, MAX_DROT_RAD)
    pos_tgt = cur_pos + dpos
    R_tgt = Rotation.from_rotvec(drot).as_matrix() @ q_cur.as_matrix()
    rpy_tgt = Rotation.from_matrix(R_tgt).as_euler("xyz")
    clipped = not (np.allclose(dpos, raw_dpos) and np.allclose(drot, raw_drot))
    return pos_tgt, rpy_tgt, dpos, drot, clipped


def workspace_ok(pos, box) -> bool:
    lo, hi = box
    return bool(np.all(pos >= lo) and np.all(pos <= hi))


# ── policy wrapper ────────────────────────────────────────────────────


def _peek_policy_type(model_dir: str) -> str:
    """Read just the `type` field of config.json, before the config class for that
    type is necessarily registered (PreTrainedConfig.from_pretrained needs it
    registered *before* it can decode the rest of the file)."""
    from lerobot.configs.policies import CONFIG_NAME
    if Path(model_dir).is_dir():
        config_file = Path(model_dir) / CONFIG_NAME
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(repo_id=model_dir, filename=CONFIG_NAME)
    return json.loads(Path(config_file).read_text())["type"]


def _peek_film_structure(model_dir: str):
    """Auto-detect the STRUCTURAL FiLM settings (`cond`, `inject`) a checkpoint
    was actually trained with, straight from its own saved tensors: `cond`
    fixes which buffers exist (_contact_F0/_fz_tau/_seal_mean only get
    registered for channels in cond -- see film_contact.apply's new_init), and
    `inject` fixes contact_film's hidden width (state_proj.out_features=960 for
    'prefix' vs vlm_with_expert.expert_hidden_size=720 for 'suffix'/'output').
    Unlike FILM_COND/FILM_INJECT env vars (which silently mismatch if the
    operator guesses wrong), this can't drift from what's actually in the
    checkpoint. `mask_force` has NO shape footprint at all -- pure runtime
    masking behavior -- so it genuinely can't be recovered this way; default
    it from the "_mask1" repo-name convention and let FILM_MASK_FORCE override."""
    from safetensors import safe_open
    if Path(model_dir).is_dir():
        st_path = Path(model_dir) / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        st_path = hf_hub_download(repo_id=model_dir, filename="model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        keys = set(f.keys())
        cond = tuple(ch for ch, buf in (("contact", "_contact_F0"), ("fz", "_fz_tau"),
                                         ("seal", "_seal_mean"), ("dfmag", "_dfmag_tau"))
                     if any(k.endswith(buf) for k in keys))
        hidden = f.get_slice("model.contact_film.scale.2.weight").get_shape()[0]
    inject = "prefix" if hidden == 960 else "suffix"
    mask_force_default = "mask1" in Path(model_dir).name.lower()
    return cond, inject, mask_force_default, hidden


def _has_film_weights(model_dir: str) -> bool:
    """Cheap check (safetensors header only, no full tensor load) for whether a
    checkpoint has contact_film.* weights. Lets load_policy() catch a missing
    or extra --film flag with a clear error, instead of torch's silent
    non-strict 'Unexpected key(s)'/'missing key(s)' warning -- which loads
    "successfully" while dropping (or failing to fill) the FiLM weights."""
    from safetensors import safe_open
    if Path(model_dir).is_dir():
        st_path = Path(model_dir) / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        st_path = hf_hub_download(repo_id=model_dir, filename="model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        return any("contact_film" in k for k in f.keys())


def load_policy(checkpoint: Path, film: bool = False):
    from lerobot.configs.parser import load_plugin
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    # checkpoint may be a local training-output dir (outputs/<run>/checkpoints/<n>,
    # containing pretrained_model/), a local pretrained_model dir, or a HF Hub repo
    # id (e.g. "Chanho-Lee/smolvla_meanflow_0708") -- from_pretrained resolves that
    # itself. Loading through the config/policy registry (not a hardcoded
    # SmolVLAPolicy import) is what lets this run smolvla_meanflow checkpoints too.
    nested = Path(checkpoint) / "pretrained_model"
    model_dir = str(nested) if nested.exists() else str(checkpoint)
    policy_type = _peek_policy_type(model_dir)
    if policy_type != "smolvla":
        # VLA_DIR (on sys.path since module import, for convert_to_lerobot etc.)
        # contains a same-named "smolvla_meanflow/" subdirectory (the plugin's
        # project root, no top-level __init__.py) which PathFinder resolves as a
        # namespace package before ever reaching the pip -e install's meta_path
        # finder -- drop it from sys.path for this one import so the real package
        # (registering the type via load_plugin) is what actually loads.
        saved_path, sys.path[:] = sys.path[:], [p for p in sys.path if p != str(VLA_DIR)]
        try:
            load_plugin(policy_type)  # registers third-party policy types, e.g. smolvla_meanflow
        finally:
            sys.path[:] = saved_path
    cfg = PreTrainedConfig.from_pretrained(model_dir)

    if film and cfg.type != "smolvla":
        raise SystemExit(f"--film is only for smolvla checkpoints, got type={cfg.type}")
    has_film_weights = cfg.type == "smolvla" and _has_film_weights(model_dir)
    if has_film_weights and not film:
        raise SystemExit(f"{model_dir} has FiLM weights (contact_film.*) but --film was not passed "
                          f"-- add --film, or it silently loads as plain smolvla and drops them.")
    if film and not has_film_weights:
        raise SystemExit(f"--film was passed but {model_dir} has no FiLM weights (contact_film.*) "
                          f"-- this is a plain smolvla checkpoint, drop --film.")

    if film:
        # FiLM condition-conditioned policy (V1/V2): patch VLAFlowMatching BEFORE
        # from_pretrained so contact_film + buffers exist and load from the checkpoint.
        # c-hat is then computed live from the obs. Variant is train-time only, so 'v2' is
        # fine here. cond/inject are auto-detected from the checkpoint's own tensors
        # (see _peek_film_structure) -- they're structural, so this can't mismatch.
        # mask_force has no shape footprint; only FILM_MASK_FORCE / the "_mask1"
        # repo-name convention can tell us, so double-check it if in doubt.
        import os
        import film_contact
        det_cond, det_inject, det_mask_default, det_hidden = _peek_film_structure(model_dir)
        cond = (tuple(c.strip() for c in os.environ["FILM_COND"].split(",") if c.strip())
                if "FILM_COND" in os.environ else det_cond)
        inject = os.environ.get("FILM_INJECT", det_inject)
        mask_force = (os.environ["FILM_MASK_FORCE"] not in ("0", "false", "False")
                      if "FILM_MASK_FORCE" in os.environ else det_mask_default)
        print(f"[run_policy] FiLM structure from checkpoint tensors: cond={cond} "
              f"inject={inject} (contact_film hidden={det_hidden}) | mask_force={mask_force} "
              f"({'FILM_MASK_FORCE env' if 'FILM_MASK_FORCE' in os.environ else '\"_mask1\" in repo name'} "
              f"-- NOT verifiable from weights, override with FILM_MASK_FORCE=0/1 if wrong)")
        f0 = float(os.environ.get("FILM_F0", "12"))
        tau = float(os.environ.get("FILM_TAU", "10"))        # contact-DROP scale; MUST match training
        fz_tau = float(os.environ.get("FILM_FZ_TAU", "30"))  # fz scale; MUST match training
        dfmag_tau = float(os.environ.get("FILM_DFMAG_TAU", "5"))  # d|F|/dt scale; MUST match training
        # wrench/seal stats only depend on observation.state (shared by delta and
        # abs conversions of the same recordings), so either dataset variant works;
        # lges_suction (pre-0708) no longer exists on this machine. dfmag checkpoints
        # (cond incl. 'dfmag', state 16) need a *_dF dataset, e.g.
        # FILM_DATASET=lges_case_pick_0708_dF.
        ds = VLA_DIR / "datasets" / os.environ.get("FILM_DATASET", "lges_case_pick_0708")
        wm, ws = film_contact.load_wrench_stats(ds)
        sm, ss = film_contact.load_seal_stats(ds)
        dm, dsd = film_contact.load_dfmag_stats(ds)
        film_contact.apply("v2", wm, ws, seal_mean=sm, seal_std=ss, cond=cond,
                           contact_F0=f0, contact_tau=tau, fz_tau=fz_tau,
                           mask_force=mask_force, inject=inject,
                           dfmag_mean=dm, dfmag_std=dsd, dfmag_tau=dfmag_tau)
        print(f"[run_policy] FiLM ENABLED (cond={cond} inject={inject} mask_force={mask_force} "
              f"F0={f0:.0f} tau={tau:.0f} fz_tau={fz_tau:.0f} dfmag_tau={dfmag_tau:.0f})")

    policy = get_policy_class(cfg.type).from_pretrained(model_dir, config=cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    print(f"[run_policy] loaded {cfg.type} checkpoint from {model_dir}")
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


def _film_diagnostics(policy) -> dict | None:
    """Summarize the live FiLM condition and gain vectors for rollout plots."""
    model = policy.model
    c = getattr(model, "_cur_contact", None)
    film = getattr(model, "contact_film", None)
    if c is None or film is None:
        return None
    p = next(film.parameters())
    c_eval = c.detach().to(device=p.device, dtype=p.dtype)
    with torch.inference_mode():
        gamma = film.scale(c_eval)[0].float().cpu().numpy()
        beta = film.shift(c_eval)[0].float().cpu().numpy()

    def stats(v):
        return {
            "mean": float(v.mean()),
            "abs_mean": float(np.abs(v).mean()),
            "rms": float(np.sqrt(np.mean(v * v))),
            "abs_max": float(np.abs(v).max()),
            "abs_p95": float(np.percentile(np.abs(v), 95)),
        }

    return {
        "cond_names": list(getattr(model, "_film_cond", ())),
        "c_hat": [float(v) for v in c[0].detach().float().cpu().tolist()],
        "gamma": stats(gamma),
        "beta": stats(beta),
    }


def latest_checkpoint() -> Path:
    runs = sorted((VLA_DIR / "outputs").glob("*/checkpoints/last"))
    if not runs:
        raise SystemExit("no checkpoint under outputs/*/checkpoints/last")
    return runs[-1]


# ── self-test (offline, no robot) ─────────────────────────────────────


def self_test(take_dir: Path, checkpoint: Path, film: bool = False):
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
                     list(j.get("right_arm", {}).values()), wrench6,
                     bool(f["suction_cmd"]), bool(f.get("vacuum_sealed")))
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
    # cm-scale errors, so 0.1 mm is a safe pass threshold.
    ok = static_pos < 1e-4
    print(f"  -> {'OK, FK path reproduces recordings (within recorder read-skew)' if ok else 'MISMATCH — investigate before robot'}\n")

    # 2. integration round-trip: integrating the recorded action onto frame t's
    #    pose should land on frame t+1's recorded pose (validates the convention).
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
    policy, pre, post = load_policy(checkpoint, film=film)
    policy.reset()
    ob.df_channel = int(policy.config.robot_state_feature.shape[0]) == 16
    abs_action = int(policy.config.action_feature.shape[0]) == 8
    suction_idx = 7 if abs_action else 6
    print(f"  action space: {'absolute pose+suction (8d)' if abs_action else 'delta pose+suction (7d)'}")
    errs = []
    n = min(len(frames) - 1, len(rgb_paths))
    import cv2
    for i in range(n):
        img = cv2.cvtColor(cv2.imread(str(rgb_paths[i])), cv2.COLOR_BGR2RGB)
        f = frames[i]
        j = f["joints"]
        w = f["wrench"]
        s = ob.state(list(j["torso"].values()), list(j["left_arm"].values()),
                     list(j.get("right_arm", {}).values()),
                     [w["fx"], w["fy"], w["fz"], w["tx"], w["ty"], w["tz"]],
                     bool(f["suction_cmd"]), bool(f.get("vacuum_sealed")))
        depth_img = (ObsBuilder.depth_image(cv2.imread(str(depth_paths[i]), cv2.IMREAD_UNCHANGED))
                     if i < len(depth_paths) else None)
        pred = predict(policy, pre, post, s, ObsBuilder.image(img), instruction, depth_img)
        # recorded action
        cur_pos, nxt_pos = np.array(f["ee"]["pos"]), np.array(frames[i + 1]["ee"]["pos"])
        if abs_action:
            rec, err = nxt_pos, pred[:3] - nxt_pos
        else:
            rec = nxt_pos - cur_pos
            err = pred[:3] - rec
        if i < 5:
            label = "abs pos" if abs_action else "dpos"
            print(f"  t={i:3d} pred {label}(mm)={pred[:3]*1000} suction={pred[suction_idx]:.2f} "
                  f"| rec {label}(mm)={rec*1000}")
        errs.append(np.abs(err))
    errs = np.array(errs)
    print(f"  mean |{'pos' if abs_action else 'dpos'} err| over take = {errs.mean()*1000:.2f} mm "
          f"(matches eval_offline scale)\n")


# ── live helpers ──────────────────────────────────────────────────────


def _force_mag(mover) -> float:
    """Live raw wrench force magnitude (N). Carries a ~14 N gravity offset."""
    ws = getattr(mover._arm, "wrench_sensor", None)
    if ws is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(ws.get_state()["wrench"], float)[:3]))


def _baseline_force(mover, n: int = 5) -> float:
    """Mean raw force over n reads — the payload-aware reference for the force
    guard. Re-taken at each task start so a held payload doesn't bias it."""
    return float(np.mean([_force_mag(mover) for _ in range(n)]))


def _retreat_to_hover(mover, cap_z: float | None = None, by: float = 0.12):
    """Back the EE straight up by `by` metres (x, y and orientation held) to
    relieve a downward contact after an abort/stall — a BOUNDED relative lift,
    NOT a move to an absolute hover (from a drifted near-reach-limit pose an
    absolute climb can cave the lateral hold and slew off-target).

    cap_z (the highest reference target z commanded this run, i.e. the hover the
    arm descended from) bounds the lift so an abort already high up can't be
    pushed past the workspace. min-motion IK (move_ee) keeps one branch."""
    pos, rpy = mover.current_ee_pose()
    target_z = float(pos[2]) + by
    if cap_z is not None:
        target_z = min(target_z, float(cap_z))
    if target_z <= float(pos[2]) + 1e-3:
        return
    print(f"  retreating: z {pos[2]:.2f} -> {target_z:.2f} (x, y held"
          f"{f', capped at prev target {float(cap_z):.2f}' if cap_z is not None else ''})")
    if mover.move_ee([float(pos[0]), float(pos[1]), target_z], tuple(rpy), quiet=True) is None:
        print("  retreat target unreachable; leaving arm in place")


def _box_from_detection(center, margin_xy: float = 0.20):
    """Workspace box = the detected case column (center ± margin) UNION the
    view-park start point, z from just under the box-floor contact to just
    above transport. Replaces the old taught-pose per-task boxes — the case
    can be anywhere the chassis parked it."""
    import LGES.ik_demo.config as ikcfg
    xs = [float(center[0]) - margin_xy, float(center[0]) + margin_xy]
    ys = [float(center[1]) - margin_xy, float(center[1]) + margin_xy]
    if ikcfg.ARM_VIEW_PARK_EE_POS is not None:
        px, py, _pz = ikcfg.ARM_VIEW_PARK_EE_POS
        xs += [px - 0.05, px + 0.05]
        ys += [py - 0.05, py + 0.05]
    lo = np.array([min(xs), min(ys), ikcfg.DESCENT_CHECK_BOTTOM_EE_Z - 0.02])
    hi = np.array([max(xs), max(ys), ikcfg.SAFE_TRANSPORT_Z + 0.06])
    return lo, hi


class RolloutLog:
    """Persist a live rollout to the recorder's states.jsonl / meta.json layout
    so Research/gradual_drift can profile its deviation. One take dir per
    sub-task (mirroring the demo recordings), named <stamp>_ep<NN>_<task>."""

    def __init__(self, root: Path, checkpoint: Path, save_images: bool = False,
                 run_num: int = 0, action_space_label: str = "ee_delta+suction"):
        self.root = Path(root)
        suffix = f"_r{run_num:02d}" if run_num > 1 else ""
        self.stamp = time.strftime("%Y%m%d-%H%M%S") + suffix
        self.checkpoint = str(checkpoint)
        self.action_space_label = action_space_label
        self.save_images = save_images
        self.f = None
        self.take_dir = None
        self.task = self.instruction = None
        self.extra = {}
        self.n = 0

    def open_task(self, ep: int, task: str, instruction: str, extra: dict | None = None):
        """`extra` (e.g. the detection center / stack height — the analysis IVs)
        is merged into the take's meta.json at close()."""
        self.close()
        self.take_dir = self.root / f"{self.stamp}_ep{ep:04d}_{task}"
        self.take_dir.mkdir(parents=True, exist_ok=True)
        self.f = (self.take_dir / "states.jsonl").open("w")
        self.task, self.instruction, self.n = task, instruction, 0
        self.peak_f, self.peak_i = 0.0, -1
        self.extra = dict(extra or {})
        if self.save_images:
            (self.take_dir / "head_rgb").mkdir(exist_ok=True)
            (self.take_dir / "head_depth").mkdir(exist_ok=True)

    def frame(self, t, state, wrench6, pred, clamped, chunk_boundary,
              rgb=None, depth_m=None, film_diag=None):
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
        fmag = float(np.linalg.norm(np.asarray(wrench6[:3], dtype=float)))
        if fmag > self.peak_f:
            self.peak_f, self.peak_i = fmag, self.n
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
            "film": film_diag,
        }) + "\n")
        self.n += 1

    def close(self, success: bool | None = None):
        if self.f is None:
            return
        self.f.close()
        self.f = None
        meta = {
            "phase": self.task, "instruction": self.instruction,
            "action_space": self.action_space_label, "ee_rotation": "quat_wxyz, base_link",
            "source": "run_policy.py rollout", "checkpoint": self.checkpoint,
            "frames": self.n, "success": success,
            "peak_force_n": round(self.peak_f, 2), "peak_force_frame": self.peak_i,
        }
        baseline = self.extra.get("baseline_force_n")
        if baseline is not None:
            meta["peak_contact_n"] = round(self.peak_f - baseline, 2)
        meta.update(self.extra)
        if self.save_images:
            meta["depth_units"] = "uint16 millimetres (0 = invalid)"
        (self.take_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        try:
            plot_path = _save_film_plot(self.take_dir)
            if plot_path is not None:
                print(f"    [log] FiLM plot: {plot_path}")
        except Exception as e:  # noqa: BLE001
            print(f"    [log] FiLM plot failed: {e}")
        peak_msg = f"    [log] peak |F| {self.peak_f:.1f}N @frame {self.peak_i}/{self.n}"
        if baseline is not None:
            peak_msg += f" (contact +{self.peak_f - baseline:.1f}N over baseline {baseline:.1f}N)"
        print(peak_msg)



def _save_film_plot(take_dir: Path):
    """Create a compact live-rollout FiLM diagnostic plot."""
    frames = [json.loads(line) for line in (take_dir / "states.jsonl").open()]
    frames = [frame for frame in frames if frame.get("film") is not None]
    if not frames:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((take_dir / "meta.json").read_text())
    x = np.arange(len(frames))
    z = np.array([frame["ee"]["pos"][2] for frame in frames])
    top_z = meta.get("top_face_z")
    height = (z - top_z) * 100 if top_z is not None else z * 100
    height_label = "clearance above detected top (cm)" if top_z is not None else "EE z (cm)"
    pred_z = np.array([frame["action_pred"][2] for frame in frames])
    if meta.get("action_space", "").startswith("ee_abs"):
        pred_dz = (pred_z - z) * 1000
    else:
        pred_dz = pred_z * 1000
    cmd_dz = np.array([frame["action_cmd"][2] for frame in frames]) * 1000
    force = np.array([np.linalg.norm([frame["wrench"][k] for k in ("fx", "fy", "fz")])
                      for frame in frames])
    baseline = meta.get("baseline_force_n", 0.0)
    seal = np.array([frame["vacuum_sealed"] for frame in frames], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(x, height, label=height_label, color="tab:blue")
    ax.set_ylabel("height (cm)")
    ax2 = ax.twinx()
    ax2.plot(x, pred_dz, label="predicted dz", color="tab:orange", alpha=0.75)
    ax2.plot(x, cmd_dz, label="command dz", color="tab:red", alpha=0.75)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("dz (mm/tick)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[1].plot(x, force, label="|F|", color="tab:purple")
    axes[1].plot(x, force - baseline, label="contact above baseline", color="tab:brown")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("force (N)")
    axes[1].legend(loc="upper left")
    seal_ax = axes[1].twinx()
    seal_ax.step(x, seal, where="post", label="seal", color="tab:green", alpha=0.7)
    seal_ax.set_ylim(-0.05, 1.15)
    seal_ax.set_ylabel("seal")

    first_film = frames[0]["film"]
    cond_names = first_film.get("cond_names", [])
    c_hat = np.array([frame["film"]["c_hat"] for frame in frames])
    for i, name in enumerate(cond_names):
        axes[2].plot(x, c_hat[:, i], label=f"c_hat:{name}")
    axes[2].set_ylabel("FiLM input")
    axes[2].legend(loc="upper left", ncols=max(1, min(4, len(cond_names))))

    for kind, color in (("gamma", "tab:blue"), ("beta", "tab:orange")):
        rms = [frame["film"][kind]["rms"] for frame in frames]
        p95 = [frame["film"][kind]["abs_p95"] for frame in frames]
        axes[3].plot(x, rms, label=f"{kind} RMS", color=color)
        axes[3].plot(x, p95, label=f"{kind} |.| p95", color=color, linestyle="--", alpha=0.7)
    axes[3].set_ylabel("modulation")
    axes[3].set_xlabel("policy tick")
    axes[3].legend(loc="upper left", ncols=2)

    for boundary in [i for i, frame in enumerate(frames) if frame["chunk_boundary"]]:
        for ax in axes:
            ax.axvline(boundary, color="0.85", linewidth=0.5, zorder=0)
    fig.suptitle(f"FiLM rollout diagnostics: {take_dir.name}")
    out = take_dir / "film_diagnostics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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


# ── live loop ─────────────────────────────────────────────────────────


def run_live(checkpoint: Path, tasks: list[str], *, commit: bool,
             home: bool = True, force_limit_n: float | None = None,
             max_ticks: int = 1000, enforce_box: bool = False,
             log_dir: Path | None = None, log_images: bool = False,
             n_action_steps: int | None = None, loop: bool = False,
             descend_until_contact: bool = False, contact_n: float = 3.0,
             descend_floor: float = 0.74, descend_rate: float = 0.0005,
             film: bool = False, layers: int | None = None):
    """Run suction sub-task(s) at ~15 Hz on the ik_demo stack.

    commit=False -> DRY-RUN (prints, commands nothing).
    commit=True  -> COMMANDS the left arm + suction, fully guarded.

    Each run: home both arms -> view park (the collection episodes' start pose)
    -> BEV-detect the case (workspace box + logged IVs; the policy itself is
    vision-driven and does not consume the detection) -> policy loop until the
    task's done-signal (pick = seal + lift back to hover), a safety abort, or
    the tick cap.
    """
    mode = "GO — COMMANDS THE LEFT ARM" if commit else "DRY-RUN — policy commands nothing"
    print(f"{mode}. sequence: {' -> '.join(tasks)}\ncheckpoint: {checkpoint}")
    from dexcontrol.robot import Robot
    from dexcontrol.core.config import get_robot_config
    from LGES.ik_demo import config as ikcfg
    from LGES.ik_demo.suction import SuctionMover
    from LGES.ik_demo.drivers import suction_io
    from LGES.ik_demo.go_home import both_arms_home
    from LGES.ik_demo.chassis_sequence import (detect, _center_from_det, _view_park,
                                               set_head_pitch)

    ob = ObsBuilder()
    policy, pre, post = load_policy(checkpoint, film=film)
    ob.df_channel = int(policy.config.robot_state_feature.shape[0]) == 16
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
    chunk_steps = int(getattr(policy.config, "n_action_steps", 1))
    layers = int(ikcfg.SRC_LAYERS_REMAINING if layers is None else layers)
    # abs-action checkpoints (convert_to_lerobot.py --action-space abs) predict
    # the ABSOLUTE next EE pose [x,y,z,qw,qx,qy,qz,suction] instead of a delta
    # -- no running reference to integrate onto, just a per-tick safety clamp.
    abs_action = int(policy.config.action_feature.shape[0]) == 8
    suction_idx = 7 if abs_action else 6
    print(f"[run_policy] action space: {'absolute pose+suction (8d)' if abs_action else 'delta pose+suction (7d)'}")

    robot_configs = get_robot_config()
    robot_configs.enable_sensor("head_camera")
    robot_configs.sensors["head_camera"].transport = "zenoh"
    flim = ikcfg.FORCE_HARD_LIMIT_N if force_limit_n is None else force_limit_n
    with Robot(configs=robot_configs) as bot, SuctionMover(bot) as mover:
        if not bot.sensors.head_camera.wait_for_active(timeout=5.0):
            print("  (head camera may not be active)")
        set_head_pitch(bot, angle=30.0)  # BEV homography + training view

        release = mover.software_estop_active()
        if release and input("Release software E-Stop? [y/N]: ").strip().lower() != "y":
            return
        if not mover.ensure_ready(release_estop=release):
            print("arm not ready — aborting")
            return

        if commit:
            print(f"\n  *** GO COMMANDS THE LEFT ARM. Keep a hand on the e-stop. ***\n"
                  f"  clamps dpos<={(MAX_DPOS_M*1000).round(0)} mm / "
                  f"drot<={MAX_DROT_RAD*1000:.0f} mrad | force abort +{flim:.0f}N | "
                  f"workspace box {'ON' if enforce_box else 'OFF'} | "
                  f"<= {max_ticks} ticks/task"
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

            # Start exactly like a collection episode: home -> view park.
            if home:
                both_arms_home(bot, left=mover)
            _view_park(mover, "policy")

            # Detect the case: dynamic workspace box + the logged IVs (center /
            # top_face_z / layers) for by-height analysis. The policy does NOT
            # consume this — it must find the case visually.
            det = detect(bot, layers)
            det_extra = {"layers_remaining": layers}
            box = None
            if det is not None and det.found:
                center = _center_from_det(det)
                box = _box_from_detection(center)
                det_extra.update({
                    "detected_center_xyzyaw": [float(v) for v in center],
                    "top_face_z": float(det.top_face_z),
                    "detect_conf": float(det.conf),
                })
                print(f"  case @ xy=({center[0]:.3f},{center[1]:+.3f}) "
                      f"top_face_z={det.top_face_z:.4f} conf={det.conf:.2f} | "
                      f"box {box[0].round(2)}..{box[1].round(2)}")
            else:
                print("  (no case detected — box guard unavailable; policy runs blind)")
                if enforce_box:
                    print("  --box requires a detection; stopping this run.")
                    if not loop:
                        break
                    try:
                        input("\n>>> Press Enter to run again (Ctrl-C to stop) <<< ")
                    except (KeyboardInterrupt, EOFError):
                        print("\nloop stopped.")
                        break
                    continue

            log = RolloutLog(log_dir, checkpoint, save_images=log_images, run_num=run_num,
                              action_space_label="ee_abs+suction" if abs_action else "ee_delta+suction"
                              ) if log_dir is not None else None
            # Poll stdin each tick: 'q' breaks the current run; any other line
            # is the --go ENTER-abort.
            watch_stdin = commit or loop
            stdin_attrs = None
            if watch_stdin and sys.stdin.isatty():
                stdin_attrs = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin)

            # Highest commanded reference-target z this run — the hover the arm
            # descended from. An abort retreat backs UP toward it but never above it.
            max_tgt_z = -np.inf
            run_stop = None
            try:
                for ti, task in enumerate(tasks):
                    spec = TASKS[task]
                    instruction, kind = spec["instruction"], spec["kind"]
                    policy.reset()  # clear the action-chunk queue between sub-tasks
                    baseline_f = _baseline_force(mover)
                    entry_sealed = bool(seal.is_sealed()) if seal else False
                    print(f"\n=== task {ti+1}/{len(tasks)}: {task} ({kind}) | \"{instruction}\"\n"
                          f"    baseline {baseline_f:.1f}N"
                          f"{' | entry sealed' if entry_sealed else ''} ===")

                    if log is not None:
                        log.open_task(ti, task, instruction,
                                      extra={**det_extra,
                                             "baseline_force_n": round(baseline_f, 2)})
                    task_done = None
                    has_sealed = entry_sealed   # pick: latch the grasp
                    has_released = False        # place: latch the release
                    went_low = False            # latch that the arm descended to the object
                    ref_pos = ref_quat = None   # running reference target (see below)
                    for tick in range(max_ticks):
                        t0 = time.time()
                        torso_q = bot.torso.get_joint_pos()
                        left_q = np.asarray(bot.left_arm.get_joint_pos(), dtype=float)
                        right_q = bot.right_arm.get_joint_pos()
                        ws = getattr(mover._arm, "wrench_sensor", None)
                        wrench6 = (np.asarray(ws.get_state()["wrench"], float)[:6]
                                   if ws is not None else np.zeros(6))
                        t_read = time.time()
                        rgb = _grab_rgb(bot)
                        depth_m = _grab_depth(bot)
                        suction = suction_io.is_suction_commanded_on()
                        sealed = bool(seal.is_sealed()) if seal else False
                        t_cam = time.time()

                        state = ob.state(torso_q, left_q, right_q, wrench6, suction, sealed)
                        pred = predict(policy, pre, post, state, ObsBuilder.image(rgb),
                                       instruction, ObsBuilder.depth_image(depth_m))
                        t_inf = time.time()
                        cur_pos, cur_quat = state[:3], state[3:7]
                        contact = float(np.linalg.norm(wrench6[:3])) - baseline_f
                        has_sealed = has_sealed or sealed
                        # Privileged descend-until-contact gate (diagnostic): while a pick
                        # is descending and not yet in contact/sealed, ensure the EE keeps
                        # going DOWN rather than stopping at the policy's habitual depth.
                        # Overrides ONLY the z-target (lateral/rotation/suction stay from
                        # the policy, so alignment failures aren't masked). Stops the
                        # instant contact rises or seal forms; hard-guarded by
                        # descend_floor and the force-limit abort below.
                        gating = (descend_until_contact and kind == "pick"
                                  and not has_sealed and contact < contact_n
                                  and descend_floor < cur_pos[2] < DESCEND_Z)
                        if abs_action:
                            # Each chunk step is already an ABSOLUTE target (no
                            # running reference to integrate onto) -- just cap the
                            # implied step from the live pose per tick.
                            pos_tgt, rpy_tgt, dpos, drot, clipped = clamp_abs_action(
                                pred, cur_pos, cur_quat)
                            clip = "CLIP" if clipped else "    "
                            if gating:
                                pos_tgt[2] = min(pos_tgt[2], cur_pos[2] - descend_rate)
                            clamped = np.concatenate([dpos, drot, [pred[suction_idx]]])
                        else:
                            clamped = clamp_action(pred)
                            clip = "CLIP" if not np.allclose(clamped, pred) else "    "
                            if gating:
                                clamped[2] = min(clamped[2], -descend_rate)
                            # Integrate deltas onto a running REFERENCE target, NOT the
                            # live pose: at 15 Hz the arm lags the target, so cur_pos+dpos
                            # under-advances and the motion stalls. The policy emits an
                            # open-loop chunk and only re-reads the observation at chunk
                            # boundaries (every chunk_steps ticks), so re-ground the
                            # reference to the live pose there (a small, safe backward
                            # correction) and advance it purely by the deltas in between.
                            if tick % chunk_steps == 0:
                                ref_pos, ref_quat = cur_pos.copy(), cur_quat.copy()
                            pos_tgt, rpy_tgt, R_tgt = integrate(ref_pos, ref_quat, clamped)
                            ref_pos, ref_quat = pos_tgt, _rpy_to_quat_wxyz(rpy_tgt)
                        max_tgt_z = max(max_tgt_z, float(pos_tgt[2]))
                        film_diag = _film_diagnostics(policy) if film else None
                        if log is not None:
                            log.frame(t0, state, wrench6, pred, clamped, tick % chunk_steps == 0,
                                      rgb=rgb, depth_m=depth_m, film_diag=film_diag)
                        in_box = workspace_ok(pos_tgt, box) if box is not None else True
                        sol = mover.solve_pose(pos_tgt, rpy_tgt, seed=left_q, min_motion=True)
                        ik_ok = sol.pos_err_m <= ikcfg.REACH_TOL_M
                        t_ik = time.time()
                        went_low = went_low or cur_pos[2] < DESCEND_Z
                        if entry_sealed and not sealed:
                            has_released = True
                        at_hover = cur_pos[2] >= HOVER_Z

                        # Diagnostic: c-hat the FiLM actually saw for this chunk (set in
                        # embed_prefix at the chunk boundary).
                        chat_str = ""
                        if film:
                            cc = getattr(policy.model, "_cur_contact", None)
                            if cc is not None:
                                chat_str = " c^=[" + ",".join(f"{v:.2f}" for v in cc.flatten().tolist()) + "]"
                        act_label = "abs" if abs_action else "dpos"
                        act_mm = pred[:3] * (1.0 if abs_action else 1000.0)
                        print(f"[{ti+1}.{tick:3d}] {act_label}={act_mm.round(4)} suc={pred[suction_idx]:.2f} {clip} "
                              f"{'GATE' if gating else '    '} | "
                              f"tgt={pos_tgt.round(3)} z={cur_pos[2]:.2f} "
                              f"box={('ok' if in_box else 'OUT') if (enforce_box and box is not None) else 'off'} "
                              f"ik={'ok' if ik_ok else 'best'} contact={contact:+.1f}N "
                              f"seal={'Y' if sealed else '.'}{chat_str} | "
                              f"rd={(t_read-t0)*1000:.0f} cam={(t_cam-t_read)*1000:.0f} "
                              f"inf={(t_inf-t_cam)*1000:.0f} ik={(t_ik-t_inf)*1000:.0f}ms")

                        # Task-done (success) — checked before aborts. Completes only
                        # after the grasp/release AND the lift/retract back to hover,
                        # so the case is actually lifted. Suction is NOT dropped here.
                        if kind == "pick" and has_sealed and went_low and at_hover:
                            task_done = f"grasped+lifted@{tick}"; break
                        if kind == "place" and has_released and went_low and at_hover:
                            task_done = f"placed+retracted@{tick}"; break

                        # Operator stdin (non-blocking, after task-done so a finishing
                        # task is not pre-empted): cbreak mode makes q stop immediately
                        # without Enter. Enter still aborts --go. EOF is ignored.
                        if watch_stdin and select.select([sys.stdin], [], [], 0)[0]:
                            key = sys.stdin.read(1)
                            if key.lower() == "q":
                                run_stop = "operator (q)"; break
                            if key in ("\r", "\n") and commit:
                                run_stop = "ABORT: operator (ENTER)"; break

                        if commit:
                            if contact > flim:
                                run_stop = f"ABORT: force {contact:.1f}N > +{flim:.1f}N"; break
                            if enforce_box and box is not None and not in_box:
                                run_stop = f"ABORT: target {pos_tgt.round(3)} out of the detection box"; break
                            # Command the best-effort IK solution (like the scripted ik_demo
                            # legs); non-convergence is fine. Abort only on a genuine
                            # blow-up: a large one-tick joint jump (near-singular).
                            dq = float(np.abs(sol.q - left_q).max())
                            if dq > MAX_JOINT_STEP_RAD:
                                run_stop = f"ABORT: joint jump {dq:.2f} rad > {MAX_JOINT_STEP_RAD} (IK near-singular?)"; break
                            mover._arm.set_joint_pos(sol.q)
                            # Suction commands hit a weblogic endpoint that blocks ~0.5 s
                            # per call, so command it only on a state CHANGE — otherwise
                            # the loop is pinned at ~2 Hz by the suction call alone.
                            want_suction = pred[suction_idx] > 0.5
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
                if stdin_attrs is not None:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, stdin_attrs)
                if log is not None:
                    log.close(success=False)

            if commit:
                # Clean finish leaves the arm at hover holding the object; drop
                # suction so the case releases before the next loop run starts.
                suction_io.suction_off()
                if run_stop != "sequence complete":
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
                tail = "retreated; suction off."
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--self-test", type=Path, metavar="TAKE_DIR",
                    help="offline validation against a recorded take (no robot)")
    ap.add_argument("--dry-run", action="store_true",
                    help="live print-only loop (needs robot, policy commands nothing)")
    ap.add_argument("--go", action="store_true",
                    help="live loop that COMMANDS the left arm + suction (guarded). "
                         "Starts from home -> view park, like the collection episodes.")
    ap.add_argument("--task", default=None, choices=list(TASKS),
                    help="sub-task to run (default: case_pick)")
    ap.add_argument("--layers", type=int, default=None,
                    help="stack layers for the BEV warp plane (default: "
                         "ik_demo cfg.SRC_LAYERS_REMAINING)")
    ap.add_argument("--no-home", action="store_true",
                    help="skip the both-arms homing step at startup")
    ap.add_argument("--box", action="store_true",
                    help="enforce the detection-derived workspace box (abort if a "
                         "target leaves it). Off by default; force limit + clamps still apply.")
    ap.add_argument("--force-limit", type=float, default=None,
                    help="abort if contact force exceeds baseline by this many N "
                         "(default ik_demo cfg.FORCE_HARD_LIMIT_N=20; use ~8 for first runs)")
    ap.add_argument("--max-ticks", type=int, default=1000,
                    help="per-task tick cap (episodes are ~230-275 frames)")
    ap.add_argument("--log-dir", type=Path, default=None, metavar="DIR",
                    help="persist the rollout to DIR as states.jsonl/meta.json takes "
                         "(detection IVs included) for Research analysis")
    ap.add_argument("--log-images", action="store_true",
                    help="with --log-dir, also save per-frame head_rgb/ + head_depth/ "
                         "(recorder format) so the rollout is replayable by the policy")
    ap.add_argument("--loop", action="store_true", default=True,
                    help="after each run completes (or aborts), prompt Enter to run again "
                         "(enabled by default); "
                         "press q mid-run to stop immediately and return to that prompt; "
                         "Ctrl-C exits.")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="override policy n_action_steps (chunk open-loop length). "
                         "Lower values (e.g. 5) approach closed-loop; default uses the "
                         "trained value (50).")
    ap.add_argument("--film", action="store_true",
                    help="load a FiLM contact-conditioned checkpoint (V1/V2 from train_film.sh): "
                         "patches the model so c-hat is computed live. Use with "
                         "--checkpoint outputs/film_v2/checkpoints/last and matching FILM_* envs.")
    ap.add_argument("--descend-until-contact", action="store_true",
                    help="diagnostic gate: during a pick, override the z-delta to keep the EE "
                         "descending until contact/seal (tests whether forcing condition-use "
                         "recovers under-reach). Lateral/rotation/suction stay from the policy.")
    ap.add_argument("--contact-n", type=float, default=3.0,
                    help="contact-gate stops forcing descent when contact force (baseline-"
                         "subtracted N) exceeds this (default 3.0)")
    ap.add_argument("--descend-floor", type=float, default=0.74,
                    help="contact-gate hard z-floor (m): never force descent below this "
                         "(default 0.74 = box floor + suction length)")
    ap.add_argument("--descend-rate", type=float, default=0.006,
                    help="min downward z step (m/tick) the contact-gate enforces (default 0.006)")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.self_test, args.checkpoint or latest_checkpoint(), film=args.film)
        return
    if not (args.dry_run or args.go):
        ap.error("choose --self-test TAKE_DIR, --dry-run, or --go")
    tasks = [args.task or "case_pick"]
    ckpt = args.checkpoint or latest_checkpoint()

    run_live(ckpt, tasks, commit=args.go, home=not args.no_home,
             force_limit_n=args.force_limit, max_ticks=args.max_ticks,
             enforce_box=args.box, log_dir=args.log_dir, log_images=args.log_images,
             n_action_steps=args.n_action_steps, loop=args.loop,
             descend_until_contact=args.descend_until_contact, contact_n=args.contact_n,
             descend_floor=args.descend_floor, descend_rate=args.descend_rate,
             film=args.film, layers=args.layers)


if __name__ == "__main__":
    main()
