# gradual_drift

Diagnose *how a closed-loop rollout deviates from demonstrated behavior over
time*, to separate the two failure modes of low-data imitation:

- **mode (i) — covariate-shift drift:** POSE deviation rises gradually from
  early in the rollout; the conditioning state walks off the demo manifold and
  the policy extrapolates.
- **mode (ii) — action infidelity at contact:** pose deviation stays in-band
  and a sudden spike appears at a FORCE event / chunk boundary.

## Metric

Per timestep, the k-NN distance from the rollout's 15-dim `observation.state`
to the nearest demonstrated states, z-scored on the demo statistics. Split into
feature groups — `pose` (EE position+orientation) and `force` (wrench). k-NN
(distance to the demo *manifold*), not Mahalanobis, because demos are curved
trajectories, not a blob.

The **in-distribution band** is the leave-one-out deviation of the demos
themselves (each demo scored against the others). A rollout staying inside the
band is close to demonstrated behavior; rising above it is drift.

## Data

Reads recorder `states.jsonl` takes directly (numpy only — no parquet/lerobot).
Demos: `LGES/recordings/<phase>/`. Rollouts: any take in the same format, e.g.
`Research/intervention/interventions/`.

## Use

```bash
# held-out demo = in-distribution control (what "no drift" looks like)
python Research/gradual_drift/analyze.py --phase case_pick

# a real closed-loop rollout
python Research/gradual_drift/analyze.py --phase case_pick \
    --rollout Research/intervention/interventions/intervention_case_pick/<take>
```

Outputs `out/<take>.png` (pose / force deviation + contact strip) and
`out/<take>.json` (peak, fraction of frames above the demo p95 band, first
crossing frame).

## Capturing clean rollouts

`run_policy.py --log-dir DIR` persists each sub-task as a `states.jsonl` /
`meta.json` take (same schema, plus per-frame `chunk_boundary` and
`action_pred`/`action_cmd`). The prev-action executor
(`run_policy_prevaction.py`) inherits the flag, and both log the same 15-dim
state — so baseline vs. prev-action rollouts are directly comparable:

```bash
# baseline
.../run_policy.py --go --goto-start <take> --task case_pick \
    --log-dir Research/gradual_drift/rollouts/baseline
# prev-action
.../run_policy_prevaction.py --go --goto-start <take> --task case_pick \
    --log-dir Research/gradual_drift/rollouts/prevaction

python Research/gradual_drift/analyze.py --phase case_pick \
    --rollout Research/gradual_drift/rollouts/baseline/<take>
```

Note: intervention takes (`Research/intervention/...`) are *policy + human*
corrections, so their drift understates the pure policy's — use `--log-dir`
rollouts for the clean signal.
