# Causal/Reactive Chunking for Contact-Rich VLA Manipulation

**Thesis.** A slow vision-planned action chunk should be re-grounded by *fast, cheap
exteroceptive signals* (vacuum seal, wrench) at every control tick — so that the
action executed at tick *t* is causally downstream of the contact event at tick *t*,
not frozen at chunk-issue time. We add a small **fast reactive head** that runs every
tick on proprio/contact only, on top of the frozen SmolVLA chunk planner that runs
once per chunk on vision+language. This exploits the modality-cost asymmetry of our
hardware: vision/VLM is slow and expensive; seal (DI0) and wrench are cheap and fast.

Status: **proposal**. Nothing built yet. Continues the suction VLA work
(`README.md`, `EXECUTOR_DESIGN.md`).

---

## 1. Problem (grounded in our deploy)

SmolVLA predicts a chunk of `chunk_size = 50` actions and executes `n_action_steps = 50`
of them open-loop before re-reading the observation:

- `modeling_smolvla.py:340-347` — `select_action` calls the full model **only when the
  action queue is empty**, then replays 50 queued actions. For 50 ticks (~3.3 s @15 Hz)
  the executed actions derive from a **single stale observation**.
- `run_policy.py:530` — our deploy fix re-grounds the *integration reference* to the live
  pose at chunk boundaries, but the **action deltas are still the frozen chunk**. A
  contact event mid-chunk (seal 0→1, force spike) does not influence the action until the
  next chunk boundary.

For free-space reaching this is fine (that's why chunking works). For **contact events**
it is exactly wrong: the policy planned a descent on stale vision and keeps commanding it
after contact, until re-grounding. This is a candidate root cause of the hover/creep and
clamp-sensitivity we hand-tuned around in deploy.

## 2. Why our setup is uniquely suited

- **Asymmetric modality cost.** Measured in deploy: inference ~6 ms normally but ~232 ms
  on chunk regen (full VLM); IK ~1 ms; reads cheap. Running the full model every tick
  → ~4 Hz, too slow. Hence chunking. But seal/wrench are ~free every tick. The natural
  answer to "reactivity is expensive" is a **cheap fast head**, not faster vision.
- **Free exteroceptive success signal.** `vacuum_sealed` (DI0) is logged per frame in
  every take and is a real, automatically-labeled contact event. Most VLA setups don't
  have this.
- **The reactive behavior is already in the data.** Scripted demos descend until contact
  then seal; the recorded next-state action *already* stops at contact. So a fast head can
  be trained supervised, from existing takes — no new data, no teleop.
- **Real-robot eval available.** We can measure on-hardware seal success / overshoot, not
  just offline error.

## 3. Hypothesis

> H1. Re-grounding the action (not just the integration reference) on cheap contact
> signals between chunk boundaries reduces post-contact overshoot force and improves
> seal-success rate / time-to-seal, **without** paying full-model latency every tick.
>
> H2. A learned fast reactive head beats the obvious cheap baseline (event-triggered
> full re-grounding) on smoothness and/or success — and if it does *not*, the cheap
> baseline is the honest answer and the negative result is itself the finding.

## 4. Related work & the precise gap

- **RTC — Real-Time Execution of Action Chunking Flow Policies** (arXiv:2506.07339, Physical
  Intelligence). Inference-time, no retraining: generate the next chunk while executing the
  current, freeze guaranteed-execute actions, inpaint the rest via flow guidance. Solves
  **inference-latency smoothness / async stitching**. *Already shipped in our lerobot 0.5.1*
  (`rtc_config`, `_rtc_enabled`, `modeling_smolvla.py:332`) → a free baseline.
- **Bidirectional Decoding (BID).** Sample many chunks, select by backward coherence +
  forward contrast. Solves **consistency vs reactivity via search over vision-planned chunks**.
- **FASTER / TIDAL / PD-VLA.** Throughput/latency: parallel decoding, interleaving, higher
  control frequency.

**The gap.** All of the above re-ground on **vision**, faster or smoother. None exploit a
**cheap, fast, exteroceptive channel** (seal/force) to reactively modulate a chunk that was
planned on slow vision, **triggered by contact events**. Our contribution is
*modality-asymmetric, event-driven* intra-chunk reactivity. RTC is complementary (it fixes
*how* we stitch chunks; we fix *what happens between* them when contact occurs).

## 5. Method — two-rate asymmetric-modality head

```
            once per chunk (slow, vision+lang)        every tick (fast, proprio+contact)
  obs(rgb,depth,lang) ─► SmolVLA expert ─► chunk a[0..49]    seal, wrench, q ─► FastHead ─► δ
                                       │                                          │
                                       └────────►  executed_action = combine(a[k], δ) ──► robot
```

`FastHead` is tiny (a few-layer MLP / TCN over a short proprio+contact window). It does NOT
see images. Decision forks (Section 8) — what it outputs:

1. **Correction (residual):** `executed = a[k] + δ`, δ small, contact-conditioned. Trained to
   predict the residual between the frozen chunk and the recorded next-state action.
2. **Suction gate:** override the discrete suction bit causally on seal/force (smallest,
   highest-value change — the suction decision is exactly the contact-triggered one).
3. **Re-ground trigger:** δ is a binary "the world changed, re-invoke the planner now"
   signal → **event-triggered re-grounding** instead of fixed `tick % chunk_steps`.
4. **Time-warp / velocity scale:** scale progression through the chunk (slow/stop the descent
   on contact) without changing direction.

**Training.** Frozen SmolVLA (no expensive VLM retraining — key for thin data, ~100 takes).
The fast head trains on existing takes: target = the residual/trigger that the recorded
action implies given where a frozen-chunk replay would have diverged. Tiny head + thin data
is fine *because* it only sees low-dim proprio.

## 6. Experimental plan (phased — measure before building)

- **P0 — Quantify the gap (no new code beyond analysis).** Offline, on recorded rollouts:
  re-run the model every tick and measure chunk-action divergence vs intra-chunk step *k*,
  **conditioned on distance to the next contact event**. Deploy metric: *post-contact lag*
  (ticks between seal/force event and the first action that responds) and *overshoot force*.
  **Gate:** if the gap is negligible, stop — the topic is dead and that's a real finding.
- **P1 — Cheap baselines (must try before any architecture).**
  (a) **Event-triggered re-grounding**: re-run full model when force crosses a threshold or
  seal flips, else replay. ~10-line change to the run loop. (b) Shorter `n_action_steps`.
  (c) **Turn RTC on** (config flag). These set the bar the learned head must beat.
- **P2 — The fast reactive head (the contribution).** Build one fork from Section 8 (start
  with the suction gate or re-ground trigger — smallest). Train on existing takes.
- **P3 — Evaluation.** Offline (divergence, post-contact lag) + on-robot (seal-success rate,
  time-to-seal, peak overshoot force, action smoothness) across: frozen-chunk (current) vs
  event-triggered vs RTC vs fast-head. Same `--go --goto-start <take>` protocol as today.

## 7. Metrics

- **Reactivity:** post-contact lag (ticks); peak contact force after seal/contact event.
- **Task:** seal-success rate; time-to-seal; chain completion.
- **Quality:** action smoothness (jerk / chunk-boundary discontinuity); clamp-clip rate.
- **Cost:** mean/95p tick latency; full-model invocations per episode.

## 8. Open design decisions (genuinely ours to make)

1. **Fast-head output** (correction vs suction-gate vs re-ground-trigger vs time-warp).
   Recommend starting with **suction-gate or re-ground-trigger** (smallest, clearest causal story).
2. **Combine rule:** additive residual vs gating vs override.
3. **Train target derivation** for the head from existing takes (residual-to-recorded vs
   divergence-trigger labeling).
4. **Scope of first on-robot test:** `case_pick` only (as the executor work did).

## 9. Risks / what could kill it (state up front)

- **P1 wins.** Event-triggered re-grounding may capture most of the benefit; then the learned
  head is unjustified. Acceptable — it's the honest result and still a clean paper/finding.
- **Thin data** for the head — mitigated by low-dim input + frozen backbone.
- **Contact signal noise** (seal flicker, raw-wrench gravity offset ~14 N) — must baseline/
  debounce, as deploy already does.
- **Surgery surface:** the clean insertion point is the run-loop combine step
  (`run_policy.py` ~530) and/or `select_action`'s re-ground condition
  (`modeling_smolvla.py:340`), not the flow-matching expert internals — keep it surgical.

## 10. Immediate next step

Run **P0**: an offline script over existing takes that quantifies the open-loop chunk gap
around contact events. This is the cheap experiment that tells us whether the whole topic is
real before we build anything.
