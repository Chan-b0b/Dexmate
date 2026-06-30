#!/usr/bin/env python3
"""Audit the POSE/ACTION signals + the raw->lerobot conversion (data step B).

Reuses convert_to_lerobot.load_take (the canonical state/action builder) so we audit
exactly what training saw, and cross-checks the rotation convention with scipy (an
independent code path).

Three checks:
  1. ROTATION CONVENTION  drot == rotvec(R_{t+1} R_t^T)? and does integrating it back
     (exp(drot) @ R_t) recover the next pose? (validates the action def + the executor's
     R_tgt = exp(drot) @ R_cur integration).
  2. ACTION SANITY        dz targets + per-axis dpos/drot ranges, NaN/inf, quaternion
     sign-continuity in the state, suction-action == suction_cmd[t+1].
  3. CONVERSION PARITY    lerobot dataset state/action == load_take output (--parity).

  /home/dexmate/vla_venv/bin/python inspect_pose_action.py            # raw checks, all tasks
  /home/dexmate/vla_venv/bin/python inspect_pose_action.py --parity   # + lerobot parity
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import convert_to_lerobot as cvt

VLA_DIR = Path(__file__).resolve().parent
RECORDINGS = VLA_DIR.parent / "recordings"


def R_from_wxyz(q):                       # (N,4) wxyz -> scipy Rotation (xyzw)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def audit_take(take_dir):
    loaded = cvt.load_take(take_dir, with_depth=True)
    if loaded is None:
        return None
    _, _, _, states, actions = loaded
    pos, quat = states[:, 0:3].astype(np.float64), states[:, 3:7].astype(np.float64)
    suc = states[:, 7]
    dpos, drot, suc_a = actions[:, 0:3], actions[:, 3:6], actions[:, 6]
    n = len(states)

    # 1. rotation convention: converter drot vs scipy R_{t+1} R_t^-1
    Rt, Rt1 = R_from_wxyz(quat[:-1]), R_from_wxyz(quat[1:])
    drot_sp = (Rt1 * Rt.inv()).as_rotvec()
    conv_err = np.linalg.norm(drot - drot_sp, axis=1)          # should be ~0
    # integration round-trip: exp(drot) @ R_t should reproduce R_{t+1}
    Rt1_rec = Rotation.from_rotvec(drot) * Rt
    recon_ang = (Rt1_rec * Rt1.inv()).magnitude()              # rad, should be ~0
    # position round-trip (alignment check)
    pos_err = np.linalg.norm(dpos - (pos[1:] - pos[:-1]), axis=1)

    # 2. sanity
    raw_flips = int(np.sum(np.sum(quat[1:] * quat[:-1], axis=1) < 0))   # in stored state
    suc_mismatch = int(np.sum(np.abs(suc_a - suc[1:]) > 1e-6))
    bad = int(np.sum(~np.isfinite(states)) + np.sum(~np.isfinite(actions)))
    return dict(
        take=take_dir.name, n=n,
        conv_err_max=float(conv_err.max()), recon_ang_max=float(recon_ang.max()),
        pos_err_max=float(pos_err.max()), quat_flips=raw_flips,
        suc_mismatch=suc_mismatch, nonfinite=bad,
        dz_min=float(dpos[:, 2].min()), dz_max=float(dpos[:, 2].max()),
        dpos_absmax=np.abs(dpos).max(0), drot_absmax=np.abs(drot).max(0),
    )


def raw_checks(tasks):
    for task in tasks:
        tdir = RECORDINGS / task
        if not tdir.is_dir():
            continue
        rows = [audit_take(t) for t in sorted(p for p in tdir.iterdir() if p.is_dir())]
        rows = [r for r in rows if r]
        if not rows:
            continue
        print(f"\n══ {task}  ({len(rows)} takes) ═══════════════════════════════════")
        print(f"{'take':<34} {'convErr':>8} {'reconDeg':>8} {'posErr':>8} {'qFlip':>5} "
              f"{'sucBad':>6} {'NaN':>4} {'dzMin(mm)':>9} {'dzMax(mm)':>9}")
        for r in rows:
            print(f"{r['take']:<34} {r['conv_err_max']:>8.1e} "
                  f"{np.degrees(r['recon_ang_max']):>8.1e} {r['pos_err_max']:>8.1e} "
                  f"{r['quat_flips']:>5d} {r['suc_mismatch']:>6d} {r['nonfinite']:>4d} "
                  f"{r['dz_min']*1000:>9.1f} {r['dz_max']*1000:>9.1f}")
        ce = max(r["conv_err_max"] for r in rows)
        ra = np.degrees(max(r["recon_ang_max"] for r in rows))
        pe = max(r["pos_err_max"] for r in rows)
        dpa = np.max([r["dpos_absmax"] for r in rows], axis=0) * 1000
        dra = np.max([r["drot_absmax"] for r in rows], axis=0) * 1000              # rad -> mrad
        flips = sum(r["quat_flips"] for r in rows)
        sucbad = sum(r["suc_mismatch"] for r in rows)
        nan = sum(r["nonfinite"] for r in rows)
        dzmin = min(r["dz_min"] for r in rows) * 1000
        print(f"  rotation: convErr<={ce:.1e} (drot==R_t+1 R_t^T), reconErr<={ra:.1e}deg "
              f"(exp(drot)@R_t recovers pose); posErr<={pe:.1e}")
        print(f"  |dpos| max per axis = [{dpa[0]:.0f}, {dpa[1]:.0f}, {dpa[2]:.0f}] mm/step;  "
              f"max single-step descend dz = {dzmin:.0f} mm")
        print(f"  |drot| max per axis = [{dra[0]:.0f}, {dra[1]:.0f}, {dra[2]:.0f}] mrad/step")
        print(f"  quat sign-flips in state: {flips} (want 0)   suction-action mismatches: "
              f"{sucbad} (want 0)   non-finite: {nan} (want 0)")


def parity(repo_id, root, tasks, val_per_task=2):
    """Compare each lerobot episode to its source take's load_take output, byte-for-byte
    on state/action. Reconstructs the converter's deterministic take ordering."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)
    bounds = [(e["dataset_from_index"], e["dataset_to_index"]) for e in ds.meta.episodes]

    # converter order: SUCTION_TASKS order, sorted takes, drop last val_per_task, skip None
    src = []
    for task in tasks:
        tdir = RECORDINGS / task
        if not tdir.is_dir():
            continue
        takes = sorted(p for p in tdir.iterdir() if p.is_dir())
        for t in (takes[:-val_per_task] if val_per_task else takes):
            loaded = cvt.load_take(t, with_depth=True)
            if loaded is not None:
                src.append((t, loaded[3], loaded[4]))   # (dir, states, actions)

    print(f"\n══ conversion parity ({ds.num_episodes} lerobot eps vs {len(src)} source takes) ══")
    if ds.num_episodes != len(src):
        print(f"  !! COUNT MISMATCH: dataset {ds.num_episodes} episodes vs {len(src)} source takes")
    smax = amax = 0.0
    nmax = min(ds.num_episodes, len(src))
    for ep in range(nmax):
        lo, hi = bounds[ep]
        tdir, states, actions = src[ep]
        ds_state = np.stack([ds[i]["observation.state"].numpy() for i in range(lo, hi)])
        ds_act = np.stack([ds[i]["action"].numpy() for i in range(lo, hi)])
        ns = min(len(ds_state), len(states))
        sd = np.abs(ds_state[:ns] - states[:ns]).max()
        ad = np.abs(ds_act[:ns] - actions[:ns]).max()
        smax, amax = max(smax, sd), max(amax, ad)
        flag = "  <-- LEN MISMATCH" if (hi - lo) != len(actions) else ""
        if sd > 1e-5 or ad > 1e-5 or flag:
            print(f"  ep{ep:>3} {tdir.name:<34} state|Δ|={sd:.1e} act|Δ|={ad:.1e} "
                  f"(ds {hi-lo} vs raw {len(actions)} frames){flag}")
    print(f"  worst over {nmax} eps: state |Δ|={smax:.1e}  action |Δ|={amax:.1e}  (want <1e-5)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", action="store_true", help="also run lerobot conversion parity")
    ap.add_argument("--repo-id", default="local/lges_suction")
    ap.add_argument("--root", type=Path, default=VLA_DIR / "datasets/lges_suction")
    args = ap.parse_args()

    raw_checks(cvt.SUCTION_TASKS)
    if args.parity:
        parity(args.repo_id, args.root, cvt.SUCTION_TASKS)


if __name__ == "__main__":
    main()
