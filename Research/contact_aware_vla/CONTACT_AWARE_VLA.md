# Contact-Awareness in a Suction VLA — diagnosis & fix

**Origin.** While killing the reactive-chunking topic (`../reactive_chunking/`), the
P0b causal-intervention probe found that SmolVLA's action is nearly insensitive to
its own contact-state inputs: flipping `vacuum_sealed` moved the action only ~7% of a
per-step motion, the echoed `suction_cmd` bit ~8%. The policy runs ~93% on
vision + EE-pose. We call this **contact-blindness**.

**Thesis / question.** Is contact-blindness benign or harmful — and if harmful, is it
a *training-structure* or *model-structure* problem we can fix?
- **Benign reading:** vision is the true cause of the geometry; the seal/suction bits
  are redundant and lagging, so ignoring them is fine.
- **Harmful reading:** the policy can't tell it has grasped, so its *lift* is driven by
  vision/pose timing rather than the grasp event — plausibly the cause of the slow,
  hesitant, often-incomplete lift we measured in closed-loop (sealed @665 then only
  reached z=0.94, vs recorded lift to ~1.09; run1 never sealed at all).

This is squarely a model/training-structure + causal topic (the user's stated interest).

## Why it likely happens (hypotheses to test)
1. **Redundant-with-vision:** the cup-on-case geometry is visible, so SGD reads contact
   from pixels and ignores the explicit bit (shortcut learning).
2. **Low variance / imbalance:** `vacuum_sealed=1` only in the short post-seal lift
   (rare), so it carries little gradient signal.
3. **Representation dilution:** the 15-dim state enters as ONE token (`embed_prefix` in
   modeling_smolvla.py) among hundreds of image tokens — a single seal scalar is easy to
   drown out.

## Plan (offline first, like P0)
- **P1 — Characterize (offline, cheap):** per-state-dim intervention sensitivity across
  phases (approach / descent / post-seal lift). Confirms which features drive the action
  and whether seal/suction/**wrench** (untested so far) are ignored everywhere or just
  near contact. → `state_sensitivity.py`.
- **P2 — Harm test (offline):** does flipping `seal` 0→1 in the post-seal window increase
  predicted lift (dz)? Compare to the recorded lift-after-seal. Links blindness to the
  hesitant-lift behaviour.
- **P3 — Why (offline):** training-set seal/suction variance & vision-redundancy; is it
  fit-able at all (probe the state token).
- **P4 — Fix (the contribution; only if harmful):** training-structure (upweight/balance
  contact transitions, perturb seal timing, auxiliary "predict seal" loss) and/or
  model-structure (privileged contact pathway — FiLM/gate/separate token so the scalar
  isn't drowned by vision). Test whether a contact-aware policy lifts decisively / is
  more robust.

GATE: if P1/P2 show blindness is benign (vision carries contact, lift not seal-dependent
in a way that hurts), say so and stop — same honesty as the reactive-chunking gate.

Run everything with `/home/dexmate/vla_venv/bin/python`; reuse `LGES/vla_training`
(load_take, load_policy, predict_action_chunk) and `../reactive_chunking` probes as a
library. Do not modify `vla_training`.
