# Human-in-the-Loop Intervention (HG-DAgger) — Design

**Goal.** Fix the closed-loop failures we diagnosed (descend↔lift limit cycle, OOD
slow descent) by their root cause — **covariate shift**. Run the policy; the
operator takes over when it's about to fail or stalls; the human-corrected segment
is recorded in the training format and folded back into training with high weight;
iterate. This is human-gated DAgger (HG-DAgger), the standard cure for covariate
shift, and the most realistic PoC path.

**Principle:** every piece already exists — this is assembly, not new robotics.
Lives in `Research/intervention/`; reuses `LGES/vla_training` (run_policy helpers)
and `LGES/case_battery_demo` (recorder, keyboard jog, Mover) as libraries. Nothing
in those dirs is modified.

---

## 1. Architecture

A single executor `intervene.py` = run_policy's `--go` tick loop + a POLICY↔HUMAN
state machine + recording. One process, one 15 Hz loop, one keyboard owner.

```
            ┌─────────────── 15 Hz tick loop ───────────────┐
 sensors ─► read joints/wrench/rgb/depth/seal               │
            build policy obs (15- or 22-dim, model-aware)   │
            mode == POLICY:  pred = policy(obs)              │   ── command path
            mode == HUMAN :  pred = keyboard jog Δee  ───────┼──►  clamp → integrate
            safety: clamp / force-limit / box / IK-jump      │     → IK → set_joint_pos
            └────────────────────────────────────────────────┘     (+ suction on change)
 DashboardPublisher(on_sample=recorder.feed)  ── records rgb+depth+state @15 Hz
 keyboard listener thread (cbreak)            ── toggle / jog / suction / abort
```

Both modes share the **same** command path (`clamp_action → integrate → _solve_ik →
mover._arm.set_joint_pos`, run_policy.py:521-584), so the human's actions are EE
deltas recorded identically to the policy's — and to the demos.

---

## 2. Takeover hook (keyboard)

One cbreak listener thread (reuse `teach_pose._get_key`, teach_pose.py:45) owns
stdin and updates a shared `ctl` dict the loop reads. **Only one stdin owner** — we
do NOT use run_policy's `_abort_on_input` (367) or recorder's `KeyListener` (539)
concurrently (they'd fight for stdin; see the demo's pause-between caveat).

Keys (reusing `teach_pose._POS_KEYS`/`_ORI_KEYS`, teach_pose.py:40):
- **`TAB`/`i`** — toggle POLICY ↔ HUMAN.
- HUMAN jog: `w/s` ±x, `a/d` ±y, `r/f` ±z; `u/o` roll, `i/k` pitch, `j/l` yaw (each
  press = one `step_m`/`ostep` increment on the target). `x` — toggle suction.
- `+/-` jog step; `n` — mark the just-finished intervention as failed (else success);
  `ENTER`/`Ctrl-C` — stop (triggers retreat-to-hover).

On any takeover the arm holds its current pose, then follows the operator's jogged
target via the existing IK+command path (exactly teach_pose's jog, but inside the
live loop).

## 3. Control handoff & chunk-queue handback  ← critical

- **POLICY→HUMAN:** stop applying policy actions; seed the human target with the
  *live* EE pose (`mover.current_ee_pose()`); `recorder.episode_begin("intervention_<task>")`.
- **HUMAN→POLICY:** the SmolVLA action queue still holds a stale 50-step chunk
  planned *before* the takeover. Must **`policy.reset()`** so it re-plans from the
  (human-left) pose — otherwise it replays the pre-takeover plan and immediately
  re-fails. Also: re-seed the running integration reference (`ref_pos/ref_quat`) to
  the live pose, and **re-baseline force** (`_baseline_force`, run_policy.py:378) —
  the payload may have changed (e.g. case now held). `recorder.episode_end(success)`.

## 4. Intervention recording format

Reuse `RecordController` + `DashboardPublisher(on_sample=recorder.feed)` exactly as
the demo/`collect_microtasks` do (run_demo.py:138-147) → produces standard take dirs
(`head_rgb/`, `head_depth/`, `states.jsonl`, `meta.json`), **byte-compatible with
`convert_to_lerobot.py` and `convert_prevaction.py`** with no changes.

- Record **only the human segments** as takes (episode_begin on takeover →
  episode_end on handback). These are states the *policy* drove into, corrected by
  the human = exactly HG-DAgger's aggregation distribution.
- `meta.json`: `phase="intervention_<task>"`, `intervention=true`, `success` from
  the `n`-key, via `recorder.set_meta_extra` (recorder.py:129).
- Actions are recovered offline as next-state deltas of the recorded poses — same as
  demos. Works for both the 15-dim (convert_to_lerobot) and 22-dim prev-action
  (convert_prevaction adds the prev-action column) pipelines, unchanged.
- (Optional) log policy segments to a side file for review, not into training.

## 5. Retrain / weighting (DAgger aggregation)

- **Aggregate:** round *r* dataset = original demos ∪ interventions₁..ᵣ (accumulate;
  never discard).
- **Weight up the interventions** (few but high-value). Pragmatic PoC = **oversample**
  intervention episodes K×=3-5 in the converter (a `--oversample-intervention K`
  flag, or duplicate those take dirs), since lerobot has no easy per-episode weight.
- **Fine-tune** from `smolvla_base` (matched to the baseline recipe) on the aggregated
  set → deploy → collect interventions *where it still fails* → retrain. Stop when the
  takeover rate (interventions per episode) drops to ~0.

## 6. Uncertainty-gated auto-pause (phase 2, optional)

SmolVLA is flow-matching → no clean entropy. Proxy: sample N chunks via
`predict_action_chunk` with different noise, take the **dispersion** (std of the first
action across samples); above a threshold → auto-switch to HUMAN (or beep + prompt).
Reuses the model as-is. Layer on only after human-gated works.

## 7. Safety (reuse run_policy guards in BOTH modes)

Per-step clamp (`clamp_action`, MAX_DPOS_M), workspace box, **force-limit abort**
(operator can't feel force through keys → this is the real guard), IK near-singular
jump abort (>MAX_JOINT_STEP_RAD), `_retreat_to_hover` on any non-clean stop, e-stop.
Human jog steps are themselves ≤ clamp.

## 8. Reuse map (import as library; do not modify)

| need | source |
|---|---|
| policy load, obs, clamp, integrate, box, IK/Mover, goto-start, retreat, baseline-force, safety consts, TASKS | `LGES/vla_training/run_policy.py` |
| 22-dim prev-action obs (if improving the prev-action model) | `Research/contact_aware_vla/run_policy_prevaction.py` |
| keyboard cbreak + EE-jog key map | `LGES/case_battery_demo/teach_pose.py` |
| recording in take format | `dashboard/recorder.py` (RecordController) + `dashboard/publisher.py` (DashboardPublisher) |
| convert to LeRobot (+ oversample) | `convert_to_lerobot.py` / `Research/contact_aware_vla/convert_prevaction.py` |

## 9. Open decisions (for review)

1. **Takeover modality:** keyboard EE-jog (designed; reuses teach_pose; jerkier data)
   vs **gravity-comp hand-guiding** (smoother, human-quality demos — better for
   insertion; needs compliant-mode switch + handback). Recommend keyboard for v1,
   hand-guiding as the upgrade for the place/insertion tasks (ideas 3/4).
2. **Which model to improve first** — baseline (15-dim) or prev-action (22-dim). The
   executor is model-agnostic (obs dim from `policy.config`).
3. **Oversample factor K** for intervention data.
4. Record policy segments for review, or human-only?

## 10. Phased build

- **P1** — human-gated executor: keyboard toggle + jog + record human segments +
  chunk handback + safety. (`intervene.py`)
- **P2** — converter oversampling + the DAgger retrain loop.
- **P3** — uncertainty-gated auto-pause.
- **P4 (opt)** — gravity-comp hand-guiding modality for insertion-quality demos.
