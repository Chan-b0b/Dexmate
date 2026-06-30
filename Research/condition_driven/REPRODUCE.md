# Reproduce: under-reach diagnosis + contact-gate confirmation

Step-by-step to re-run the whole experiment with the **correct** checkpoint. All prior result
data was collected with the wrong weight (`smolvla_depthseal_fixed`) and has been erased; the
tooling below is unchanged. **Numbers will differ with the new weight — what you're reproducing
is the PROCEDURE and the expected PATTERN** (see §8). The new weight may or may not show the same
failure; the experiment is what tells you.

## 0. Setup
- **Pick the correct checkpoint.** It must be a dir containing `pretrained_model/`, i.e.
  `outputs/<RUN>/checkpoints/last`. **Pass it explicitly** with `--checkpoint` — do NOT rely on the
  default (`latest_checkpoint()` just takes the alphabetically-last `outputs/*/checkpoints/last`,
  which is how the wrong weight got picked).
  ```bash
  CKPT=/home/dexmate/CNS_code/Dexmate/LGES/vla_training/outputs/<RUN>/checkpoints/last
  PY=/home/dexmate/vla_venv/bin/python
  CD=/home/dexmate/CNS_code/Dexmate/Research/condition_driven   # probe lives here
  ```
- Run `run_policy.py` from `LGES/vla_training/`, the probe from `Research/condition_driven/`.
- State layout (probe relies on it): `ee_z = idx 2`, `suction_cmd = 7`, `vacuum_sealed = 8`,
  `force = 9:12`.

## 1. Verify the seal sensor (once)
```bash
$PY $CD/seal_monitor.py
```
Make/break a seal by hand → the **`DI0`** field must flip `SEAL`/`----`. (`is_sealed` stays `F`
unless suction is commanded on — that gating is expected; DI0 is the sensor truth.) This confirms
the condition signal before trusting any seal%.

## 2. Demo reference (weight-independent, but re-confirm)
Establishes ground truth: the expert is force-gated and what depth band the objects span.
```bash
cd $CD && $PY lift_condition_probe.py            # demos vs (no rollouts yet) — read the DEMO rows
```
Note the **demo deepest_z band** and the force-gated lift signature (expert |F|@lift tight, CV low).

## 3. Collect the ungated depletion sweep  ← NEW WEIGHT
Deplete the stack **top-first**; one `--log-dir` per layer, named `case_pick_<N>` where
**N=0 is the top (highest target), higher N = lower target**. ~10 rounds/layer, all layers until the
arm can't reach.
```bash
cd /home/dexmate/CNS_code/Dexmate/LGES/vla_training
$PY run_policy.py --go --task case_pick \
  --goto-start <a recorded case_pick take_dir> \
  --checkpoint $CKPT \
  --log-dir $CD/rollouts/case_pick_0 \
  --loop
# repeat per layer: case_pick_1, case_pick_2, ... (remove the top layer between runs)
```
`--loop` = home → goto-start → run, model/robot stay loaded, suction off each run, one timestamped
take dir per round (`_rNN` suffix from run 2).

## 4. Analyze the sweep
```bash
cd $CD
$PY lift_condition_probe.py --by-height rollouts/case_pick_*
```
Reports, per layer: **sealed%**, **dz(seal)** vs **dz(unseal)**, the **under-reach gap**
(= dz(unseal) − dz(seal)), against the demo band.
- **Under-reach** = positive gap (failed episodes stop *shallower* than successful ones) → didn't
  descend enough = condition not used.
- **At-depth failure** (gap ≈ 0, e.g. an alignment layer) = reached the object, didn't seal.

Also run the coherence/timing view per layer (over-press, oscillation, lift-without-seal, suction
chatter), and timelines for failing deep episodes:
```bash
$PY lift_condition_probe.py --rollouts rollouts/case_pick_4 --dump
$PY lift_condition_probe.py --timeline rollouts/case_pick_4/*     # -> timelines/*.png
```

## 5. Rule out confounds (same as before)
- **Seal sensor**: §1 (done).
- **Suction command**: every unsealed episode must have commanded suction (else the failure is "never
  tried", not under-reach). Check: `(states[:,7] > 0.5).any()` per unsealed take.
- **Chunk latency**: re-run the sweep with `--n-action-steps 5` (near closed-loop). If under-reach
  persists, it's the learned depth target, not the 50-step open-loop chunk.
- **Force channel**: in a deep-layer timeline, under-reach shows as `|F|` never rising to the demo
  contact level (stuck at the gravity baseline) = sucking air — corroborates seal-independently.

## 6. Confirm OOD is ruled out
The deep-layer object depths must be **within the demo deepest_z band** (§2). If a layer's object is
below the deepest demo, that layer is partly out-of-distribution (coverage gap), not pure
condition-ignoring — note it.

## 7. Causal confirmation — the contact gate  ← NEW WEIGHT
Run the privileged gate on the **deep layers** (where under-reach appears). It forces descent until
contact/seal; lateral/rotation/suction stay from the policy; guarded by `--descend-floor` and the
force-limit abort. **e-stop ready for the first episode.**
```bash
cd /home/dexmate/CNS_code/Dexmate/LGES/vla_training
$PY run_policy.py --go --task case_pick \
  --goto-start <take_dir> --checkpoint $CKPT \
  --descend-until-contact --force-limit 8 \
  --contact-n 3.0 --descend-floor 0.76 --descend-rate 0.006 \
  --log-dir $CD/rollouts_gated/case_pick_3 \
  --loop
# repeat for the other failing layer(s): case_pick_4, ...
```
Analyze the gated runs (same level names so the parser groups them):
```bash
cd $CD && $PY lift_condition_probe.py --by-height rollouts_gated/case_pick_*
$PY lift_condition_probe.py --timeline rollouts_gated/case_pick_4/*
```

## 8. Expected PATTERN to reproduce (numbers will differ)
1. **Ungated**: seal% **collapses as the stack lowers**; deep-layer failures **under-reach**
   (dz(unseal) > dz(seal); failures pin at a habitual depth ≈ the demo-median contact depth).
2. **Gate**: forcing descend-until-contact **recovers the deep layers** (seal% jumps toward ~100%),
   reaching the true object depths — proving the failure was **condition-ignoring**, not capability /
   data / OOD. (Gate over-presses; that motivates the learned graded fix, it is not the fix.)
3. If the new weight does **not** under-reach → that itself is the finding (the wrong-weight result
   was an artifact); re-scope before the learned method.

## Tooling reference (unchanged, weight-independent)
- `lift_condition_probe.py` — `--by-height` (per-layer under-reach + demo band), default
  (`--rollouts` coherence/timing), `--timeline`. Layer parsed from `case_pick_<N>` dir name.
- `seal_monitor.py` — live DI0 / is_sealed / suction_cmd / toolA.
- `run_policy.py` flags added for this: `--loop`, `--n-action-steps`, `--descend-until-contact`
  (+ `--contact-n`, `--descend-floor`, `--descend-rate`).
