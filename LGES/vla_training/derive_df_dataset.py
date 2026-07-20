#!/usr/bin/env python3
"""Derive a dF dataset: append dfmag = |F[:3]|_t - |F[:3]|_{t-1} (N/frame, 0 at episode
start) as observation.state dim 15 (state 15 -> 16). Contact shows as a sharp NEGATIVE
dfmag transient (~-5..-10 N/frame at 15fps), robust to the payload-dependent baseline
that breaks the fixed-F0 'contact' channel on place tasks.

  python derive_df_dataset.py datasets/lges_case_pick_0708 datasets/lges_case_pick_0708_dF

Copies data parquets with the extended state, patches meta/info.json, meta/stats.json and
the per-episode stats columns in meta/episodes/*.parquet. Images (embedded PNG bytes in
the parquets) are carried through unchanged.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WRENCH_LO, WRENCH_HI = 9, 12  # fx,fy,fz
QS = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def stats_for(x: np.ndarray) -> dict:
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std": float(x.std()), "count": int(x.size),
            **{k: float(np.quantile(x, q)) for k, q in QS.items()}}


def main(src: Path, dst: Path):
    assert (src / "meta/info.json").exists(), f"not a lerobot dataset: {src}"
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")

    all_df, per_ep = [], {}
    for pq in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(pq)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # (T,15)
        assert st.shape[1] == 15, f"{pq}: state dim {st.shape[1]} != 15"
        dfmag = np.zeros(len(df), dtype=np.float32)
        for ep in df["episode_index"].unique():                  # diff WITHIN episodes
            m = (df["episode_index"] == ep).to_numpy()
            fmag = np.linalg.norm(st[m, WRENCH_LO:WRENCH_HI], axis=1)
            d = np.diff(fmag, prepend=fmag[0])                   # first frame -> 0
            dfmag[m] = d
            per_ep.setdefault(int(ep), []).append(d)
        new_state = np.concatenate([st, dfmag[:, None]], axis=1)
        df["observation.state"] = list(new_state)
        out = dst / "data" / pq.relative_to(src / "data")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        all_df.append(dfmag)
    all_dfmag = np.concatenate(all_df)

    # meta/info.json: state 15 -> 16
    info_p = dst / "meta/info.json"
    info = json.loads(info_p.read_text())
    f = info["features"]["observation.state"]
    f["shape"] = [16]
    f["names"] = list(f["names"]) + ["dfmag"]
    info_p.write_text(json.dumps(info, indent=4))

    # meta/stats.json: append dfmag stats to observation.state arrays
    st_p = dst / "meta/stats.json"
    stats = json.loads(st_p.read_text())
    g = stats_for(all_dfmag)
    s = stats["observation.state"]
    for k in list(s.keys()):
        if k == "count":
            continue
        s[k] = list(s[k]) + [g[k]]
    st_p.write_text(json.dumps(stats, indent=4))

    # meta/episodes/*.parquet: extend per-episode observation.state stats arrays
    for eppq in sorted((dst / "meta/episodes").rglob("*.parquet")):
        ep = pd.read_parquet(eppq)
        cols = [c for c in ep.columns if c.startswith("stats/observation.state/")]
        for i, row in ep.iterrows():
            d = np.concatenate(per_ep[int(row["episode_index"])])
            eg = stats_for(d)
            for c in cols:
                key = c.rsplit("/", 1)[1]
                if key == "count":
                    continue
                ep.at[i, c] = np.array(list(np.asarray(row[c]).ravel()) + [eg[key]],
                                       dtype=np.asarray(row[c]).dtype)
        ep.to_parquet(eppq, index=False)

    print(f"[derive] {src.name} -> {dst.name}: frames={all_dfmag.size} "
          f"dfmag mean={all_dfmag.mean():.3f} std={all_dfmag.std():.3f} "
          f"min={all_dfmag.min():.1f} max={all_dfmag.max():.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
